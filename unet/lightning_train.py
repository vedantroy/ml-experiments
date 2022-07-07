from pathlib import Path
import argparse

import signal
import sys
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from model import UNet
import wandb
from data import BasicDataset

from loss import dice_loss
from utils import convert_mask_to_ground_truth


class EarlyStop(RuntimeError):
    pass


# [[1, 2, 3], [4, 5, 6]] => [[1, 4], [2, 5], [3, 6]]
def stack(arr):
    row_len = len(arr[0])
    cols = [[] for _ in range(row_len)]
    for x in arr:
        for idx, col in enumerate(cols):
            col.append(x[idx])
    return tuple(cols)


def get_losses(batch, model, n_classes):
    imgs, masks = batch["image"], batch["mask"]

    mask_preds = model(imgs)
    probs = F.softmax(mask_preds, dim=1).float()
    one_hots = convert_mask_to_ground_truth(masks, n_classes)

    loss_cross_entropy = F.cross_entropy(mask_preds, masks)
    loss_dice = dice_loss(probs, one_hots)
    combined_loss = loss_cross_entropy + loss_dice
    return (loss_cross_entropy, loss_dice, combined_loss, (imgs, masks, mask_preds))


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        scale: float,
        percent_of_data_used_for_validation: int,
        batch_size: int,
        run_id: str,
        learning_rate: int,
        **kwargs,
    ):
        super().__init__()
        self.model = UNet(n_in_channels=3, n_classes=2)
        dataset = BasicDataset(
            images_dir="./data/imgs", masks_dir="./data/masks", scale=scale
        )
        n_val = int(len(dataset) * (percent_of_data_used_for_validation / 100))
        n_train = len(dataset) - int(n_val)
        assert n_train + n_val == len(dataset)

        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            # if we use seed_everything, do we still need this?
            # generator=torch.Generator().manual_seed(0)
        )
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.loader_args = dict(
            batch_size=batch_size,
            num_workers=4,
            # Only useful if you're training on GPU
            pin_memory=self.device.type == "cuda",
        )
        self.sample_arg = None
        self.run_id = run_id
        self.learning_rate = learning_rate
        self.early_exit = False
        self.training_started = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # no LR schedulers
        return [optimizer], []

    def on_train_batch_start(self, batch, batch_idx: int, unused: int = 0):
        if self.early_exit:
            raise EarlyStop()

    def training_step(self, batch, batch_idx):
        self.training_started = True
        cross_entropy, dice, combined, _ = get_losses(batch, self, self.model.n_classes)
        wandb.log(
            {
                "loss_cross_entropy": cross_entropy,
                "loss_dice": dice,
                "loss_combined": combined,
                "step": trainer.global_step,
                "epoch": trainer.current_epoch,
            }
        )
        return combined

    def validation_step(self, batch, batch_idx):
        cross_entropy, dice, combined, (imgs, masks, mask_preds) = get_losses(
            batch, self, self.model.n_classes
        )
        if self.sample_arg == None:
            self.sample_arg = imgs

        if not self.training_started:
            # For some reason PL calls validation twice before anything has really happened
            return

        first_batch = batch_idx == 0
        return (
            cross_entropy,
            dice,
            combined,
            (imgs, masks, mask_preds) if first_batch else None,
        )

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return

        cross_entropy, dice, combined, batch_samples = stack(outputs)
        cross_entropy = torch.stack(cross_entropy).mean()
        dice = torch.stack(dice).mean()
        combined = torch.stack(combined).mean()

        (imgs, masks, mask_preds) = batch_samples[0]

        wandb.log(
            {
                "validation_loss_cross_entropy": cross_entropy,
                "validation_loss_dice": dice,
                "validation_loss_combined": combined,
                "step": trainer.global_step,
                "epoch": trainer.current_epoch,
            }
        )
        wandb.log(
            {
                "image": wandb.Image(imgs[0].cpu()),
                "masks": {
                    "true": wandb.Image(masks[0].float().cpu()),
                    "pred": wandb.Image(mask_preds.argmax(dim=1)[0].float().cpu()),
                },
                "step": trainer.global_step,
            }
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_args)

    def val_dataloader(self):
        loader = DataLoader(
            self.val_set, shuffle=False, drop_last=True, **self.loader_args
        )
        assert (
            len(loader) > 0
        ), "validation loader must have enough data for at least 1 full batch"
        return loader

    def on_train_end(self):
        # This should never happen??
        if self.early_exit:
            return

        path = Path(f"./runs/{self.run_id}")
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "final_model.pt"
        model_onnx_path = path / "final_model.onnx"
        torch.save(self.state_dict(), model_path)
        torch.onnx.export(self.model, self.sample_arg, model_onnx_path)
        wandb.save(str(model_path))
        wandb.save(str(model_onnx_path))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--id",
        "-I",
        type=str,
        default=False,
        help="Load model using an existing run id",
    )
    return parser.parse_args()


if __name__ == "__main__":
    model = run_id = config = None
    args = get_args()
    is_resuming = False
    if args.id:
        is_resuming = True
        run_id = args.id
        print(f"Resume run with id: {run_id} from checkpoint")
        lighting_checkpoint = Path(f"./runs/{run_id}/lightning.checkpoint")
        wandb_checkpoint = Path(f"./runs/{run_id}/wandb.checkpoint")
        if not lighting_checkpoint.is_file():
            raise FileNotFoundError(
                f"Could not find checkpoint at {lighting_checkpoint}"
            )
        config = torch.load(wandb_checkpoint)
        model = LightningModel.load_from_checkpoint(
            lighting_checkpoint, run_id=run_id, **config
        )
    else:
        run_id = wandb.util.generate_id()
        print(f"Starting new run with id: {run_id}")
        config = dict(
            scale=0.5,
            percent_of_data_used_for_validation=10,
            batch_size=1,
            learning_rate=3e-4,
            amp=False,
            epochs=5,
        )

    p = Path(f"./runs/{run_id}")
    p.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["epochs"],
        enable_checkpointing=False,
    )
    model = LightningModel(**config, run_id=run_id)

    with wandb.init(
        project="UNet",
        id=run_id,
        resume="must" if is_resuming else "never",
        config=config,
    ):
        wandb.watch(model.model, criterion=None, log="all", log_freq=100)

        def signal_handler(signal, frame):
            model.early_exit = True

        signal.signal(signal.SIGINT, signal_handler)
        try:
            trainer.fit(model)
        except EarlyStop:
            print("Trainer was stopped by signal handler")
            pass

    print("Saving PL checkpoint")
    lightning_checkpoint = p / "lightning.checkpoint"
    trainer.save_checkpoint(lightning_checkpoint)
    torch.save(model.model.state_dict(), p / "model.pt")

    if os.environ.get("WANDB_MODE") != "disabled":
        print("Saving WANDB checkpoint")
        torch.save(config, p / "wandb.checkpoint")

    if model.early_exit:
        print("Received SIGINT, exiting ...")
        sys.exit(1)
