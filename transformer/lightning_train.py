from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import wandb

from template import stack_apply, EarlyStop, LightningTemplate, run
from model import Transformer

class BasicDataset(Dataset):
    def __init__(self, file_name: str, sequence_len: int, vocab_size: int):
        p = Path(file_name)
        assert p.is_file(), f"File: {file_name} did not exist"
        self.path = p

        self.vocab_size = vocab_size
        self.crop_len = sequence_len + 1

        with open(self.path, "r") as f:
            corpus = f.read()

        if not corpus.isascii():
            raise ValueError("Loaded corpus is not ASCII.")

        if "\0" in corpus:
            # Reserve 0 codepoint for pad token.
            raise ValueError("Corpus must not contain null byte.")

        # Tokenize by taking ASCII codepoints.
        corpus = [ord(c) for c in corpus]
        num_seqs, leftover = divmod(len(corpus), self.crop_len)
        corpus = corpus[:-leftover]

        tensor = torch.IntTensor(corpus)
        assert tensor.max() > 0
        assert tensor.min() < self.vocab_size
        self.tensor = tensor.reshape((num_seqs, self.crop_len))

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        seq = self.tensor[idx]
        return dict(obs=seq[:-1], target=seq[1:])


def get_losses(batch, model, vocab_size):
    x, y = batch["obs"], batch["target"]
    logits = model(x)
    one_hots = F.one_hot(y.to(torch.int64), num_classes=vocab_size)
    assert logits.shape == one_hots.shape
    return F.cross_entropy(logits, one_hots.float()), (x, y)


class LightningModel(LightningTemplate):
    def __init__(
        self,
        batch_size,
        learning_rate: int,
        sequence_len: int,
        vocab_size: int,
        d_model: int,
        widening_factor: int,
        layers: int,
        heads: int,
        dropout:  float,
        **kwargs,
    ):
        super().__init__(
            model=Transformer(
                vocab_size=vocab_size,
                num_heads=heads,
                d_model=d_model,
                widening_factor=widening_factor,
                sequence_len=sequence_len,
                layers=layers,
                mask=torch.tril(torch.ones((sequence_len, sequence_len))),
                params={
                    "batch_size": batch_size,
                    "sequence_len": sequence_len,
                    "d_model": d_model,
                    "dropout": dropout,
                },
            ),
            **kwargs,
        )
        self.learning_rate = learning_rate

        self.vocab_size = vocab_size

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]
        #return {
        #    "optimizer": optimizer,
        #}

    def training_step(self, batch, batch_idx):
        super().training_step(batch, batch_idx)
        loss, (x, _) = get_losses(batch, self, self.vocab_size)
        if self.sample_arg == None:
            self.sample_arg = x
        wandb.log(
            {
                "loss": loss,
                "step": self.training_batches_seen,
                "epoch": self.trainer.current_epoch,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Don't do logging in here, since `self.training_batches_seen` won't update
        loss, (x, y) = get_losses(batch, self, self.vocab_size)

        if not self.training_started:
            # For some reason PL calls validation twice before anything has really happened
            return

        return (loss, (x, y) if batch_idx == 0 else None)

    def validation_epoch_end(self, outputs) -> None:
        if len(outputs) == 0:
            return

        loss, (x, y) = stack_apply(
            outputs, {(0,): lambda x: torch.stack(x).mean(), (1,): lambda x: x[0]}
        )

        wandb.log(
            {
                "validation_loss": loss,
                "step": self.training_batches_seen,
                "epoch": self.trainer.current_epoch,
                "learning_rate": self.lr_schedulers().get_last_lr()[0]
            }
        )

if __name__ == "__main__":
    run(
        description="Train the transformer on Shakespeare",
        project_name="Shakespeare Transformer",
        ModelClass=LightningModel,
        default_config=dict(
            val_percent=10,
            batch_size=64,
            learning_rate=5e-4,
            epochs=5000,
            vocab_size=128,
            heads=8,
            d_model=512,
            widening_factor=4,
            sequence_len=64,
            layers=6,
            dropout=0.1,
            amp=False,
            loader_args=dict(
                num_workers=4,
                batch_size=64,
            )
        ),
        trainer_args=dict(accelerator="gpu", devices=1, val_check_interval=1.0, gradient_clip_val=1.0),
        get_dataset=lambda config: BasicDataset(
            file_name="./data/shakespeare.txt",
            sequence_len=config["sequence_len"],
            vocab_size=config["vocab_size"],
        ),
        export_on_interrupt=True,
    )
