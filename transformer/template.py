from pathlib import Path
import argparse
import signal
import os
import sys
from importlib_metadata import metadata

import wandb
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class EarlyStop(RuntimeError):
    pass


# [[1, 2, 3], [4, 5, 6]] => [[1, 4], [2, 5], [3, 6]]
def stack(arr):
    row_len = len(arr[0])
    cols = [[] for _ in range(row_len)]
    for x in arr:
        for idx, col in enumerate(cols):
            col.append(x[idx])
    return cols


def stack_apply(arr, idxs_to_func):
    results = stack(arr)
    for (k, v) in idxs_to_func.items():
        for idx in k:
            results[idx] = v(results[idx])
    return tuple(results)

def save_final_files(root_path, model, sample_arg, training_batches_seen):
        # For some reason the model gets transferred off of the GPU?
        device = torch.device("cpu")
        assert sample_arg.is_cuda
        model_path = root_path / f"final_model-{training_batches_seen}.pt"
        model_onnx_path = root_path / f"final_model-{training_batches_seen}.onnx"
        torch.save(model.state_dict(), model_path)
        torch.onnx.export(model.to(device), sample_arg.to(device), model_onnx_path)
        wandb.save(str(model_path))
        wandb.save(str(model_onnx_path))


class LightningTemplate(pl.LightningModule):
    def __init__(
        self,
        train_ds,
        val_ds,
        model,
        run_id,
        loader_args,
        training_batches_seen,
        **kwargs,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model = model

        self.sample_arg = None
        self.early_exit = False
        self.training_started = False
        self.run_id = run_id
        self.loader_args = {**loader_args, "pin_memory": self.device.type == "cuda"}
        self.training_batches_seen = training_batches_seen

    def train_dataloader(self):
        loader = DataLoader(
            self.train_ds, drop_last=True, shuffle=True, **self.loader_args
        )
        assert (
            len(loader) > 0
        ), "train loader must have enough data for at least 1 full batch"
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_ds, drop_last=True, shuffle=False, **self.loader_args
        )
        assert (
            len(loader) > 0
        ), "validation loader must have enough data for at least 1 full batch"
        return loader

    def forward(self, x):
        return self.model(x)

    # Override this method & call super
    def training_step(self, batch, batch_idx):
        self.training_batches_seen += 1
        self.training_started = True
        if batch_idx > 0:
            assert (
                self.sample_arg != None
            ), f"Missing sample arg even though batch idx was: {batch_idx}"

    def on_train_batch_start(self, batch, batch_idx: int, unused: int = 0):
        if self.early_exit:
            raise EarlyStop()

def get_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--id",
        "-I",
        type=str,
        default=False,
        help="Load model using an existing run id",
    )
    return parser.parse_args()


def check_config(config):
    assert "val_percent" in config
    assert "batch_size" in config
    assert "amp" in config
    assert "epochs" in config


def check_trainer_args(trainer_args):
    assert "accelerator" in trainer_args
    assert "devices" in trainer_args

def split_dataset(dataset, val_percent):
    n_val = int(len(dataset) * (val_percent / 100))
    n_train = len(dataset) - int(n_val)
    assert n_train + n_val == len(dataset)

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        # if we use seed_everything, do we still need this?
        # generator=torch.Generator().manual_seed(0)
    )
    return train_set, val_set


def run(
    description, project_name, ModelClass, default_config, trainer_args, get_dataset, export_on_interrupt
):
    check_config(default_config)
    check_trainer_args(trainer_args)
    args = get_args(description=description)
    pl.seed_everything(42)

    model = run_id = config = None
    is_resuming = False
    if args.id:
        is_resuming = True
        run_id = args.id
        print(f"Resuming run with id: {run_id} from checkpoint")
        p = Path(f"./runs/{run_id}")
        metadata_checkpoint = Path(p / "metadata.checkpoint")
        lighting_checkpoint = Path(p / "lightning.checkpoint")
        wandb_checkpoint = Path(p / "wandb.checkpoint")
        if not lighting_checkpoint.is_file():
            raise FileNotFoundError(
                f"Could not find checkpoint at {lighting_checkpoint}"
            )
        config = torch.load(wandb_checkpoint)
        metadata = torch.load(metadata_checkpoint)
        check_config(config)
        dataset = get_dataset(config)
        train_set, val_set = split_dataset(dataset, config["val_percent"])
        model = ModelClass.load_from_checkpoint(
            lighting_checkpoint,
            **config,
            **metadata,
            run_id=run_id,
            train_ds=train_set,
            val_ds=val_set,
        )
    else:
        run_id = wandb.util.generate_id()
        print(f"Starting new run with id: {run_id}")
        config = default_config
        dataset = get_dataset(config)
        train_set, val_set = split_dataset(dataset, config["val_percent"])
        metadata = { "training_batches_seen": 0 }
        model = ModelClass(**config, **metadata, run_id=run_id, train_ds=train_set, val_ds=val_set)

    p = Path(f"./runs/{run_id}")
    p.mkdir(parents=True, exist_ok=True)

    final_trainer_args = dict(
        max_epochs=config["epochs"],
        enable_checkpointing=False,
        precision=16 if config["amp"] else 32,
        # check validation set twice per epoch
        val_check_interval=0.5,
    )
    final_trainer_args = {**final_trainer_args, **trainer_args}
    print(f"Running trainer with args: {trainer_args}")
    trainer = pl.Trainer(**trainer_args)

    interrupted = False
    with wandb.init(
        project=project_name,
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
            interrupted = True

        
        if not interrupted or export_on_interrupt:
            print("Exporting final model files...")
            save_final_files(p, model.model, model.sample_arg, model.training_batches_seen)

    print("Saving PL checkpoint")
    lightning_checkpoint = p / "lightning.checkpoint"
    trainer.save_checkpoint(lightning_checkpoint)
    torch.save(model.model.state_dict(), p / "model.pt")
    print("Saving metadata checkpoint")
    torch.save({"training_batches_seen": model.training_batches_seen}, p / "metadata.checkpoint")

    if os.environ.get("WANDB_MODE") != "disabled":
        print("Saving WANDB checkpoint")
        torch.save(config, p / "wandb.checkpoint")

    if model.early_exit:
        print("Received SIGINT, exiting ...")
        sys.exit(1)
