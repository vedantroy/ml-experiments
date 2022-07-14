import os
import argparse
from pathlib import Path
import shutil

from tqdm import tqdm
from imagen_pytorch import Imagen, BaseUnet64, ImagenTrainer
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
import wandb
from fastargs import Param, Section
from fastargs.decorators import param, get_current_config

from streaming_dataset import ImageWithCaptions

Section("files", "inputs, outputs, etc.").params(
    dataset_dir=Param(
        str, "the shards directory", default="./data/danbooru/artifacts/shards"
    ),
    checkpoint_dir=Param(str, "where to save checkpoints", default="./runs"),
    log_dir=Param(str, "where to write log files", default="./logs")
)

Section("logging", "where/when/how to log").params(
    train_log_interval=Param(
        int, "how many steps between logging training metrics", default=50
    ),
    # validation loss is always logged
)

Section("data", "how to process data").params(
    total_samples=Param(
        int, "how many samples you want to use from the dataset", default=0
    ),
    validations_per_epoch=Param(int, "how many validations to do per epoch", default=2),
    val_percent=Param(int, "percentage of samples to use for validation", default=2),
    batch_size=Param(int, "how many samples to use in each batch", default=1),
)

Section("run", "run info").params(
    run_id=Param(str, "an existing run id to use", default=""),
)


@param("files.dataset_dir")
@param("files.log_dir")
@param("data.total_samples")
@param("data.val_percent")
@param("data.batch_size")
def construct_dataloaders(dataset_dir, log_dir, total_samples, val_percent, batch_size):
    # With Mosaic, I need to pass shuffle/batch_size to the *dataset* itself
    ds = ImageWithCaptions(dataset_dir, shuffle=True, batch_size=batch_size)
    if total_samples:
        print(f"Using {total_samples} samples from the dataset")
        ds, _ = random_split(ds, [total_samples, len(ds) - total_samples])

        print(f"Writing samples to: {log_dir}")
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(exist_ok=True, parents=True)
        for idx, sample in enumerate(ds):
            torchvision.io.write_png(sample["img"], str(log_dir_path / f"{idx}.png"))
            with open(log_dir_path / f"{idx}.txt", "w+") as f: f.write(sample["tags"])

    train_ds = val_ds = None
    if val_percent == 0:
        print("Skipping validation")
        train_ds = ds
    else:
        assert val_percent > 0, "val_percent must be > 0"
        train_sz = int(len(ds) * (1 - (val_percent / 100))) if val_percent else len(ds)
        train_ds, val_ds = random_split(ds, [train_sz, len(ds) - train_sz])

    print(
        f"# train samples: {len(train_ds)}, # val samples: {len(val_ds) if val_ds else 0}"
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        # shuffling is done in the dataset
        shuffle=False,
        num_workers=12
    )

    val_dl = None
    if val_ds:
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            # We literally cannot turn off shuffling for validation :((
            shuffle=False,
            num_workers=12
        )

    return train_dl, val_dl

@param("files.checkpoint_dir")
@param("run.run_id")
def get_checkpoint_path(checkpoint_dir, run_id):
    if not run_id:
        return None

    print("Checkpoint dir: ", checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f"Checkpoint dir: {checkpoint_dir} does not exist"

    print(f"Loading state for run id: {run_id}")
    checkpoints = list((checkpoint_dir / run_id).glob("*"))

    def get_step(p):
        dir_name = str(p).split("/")[-1]
        return int(dir_name)

    checkpoints.sort(key=get_step)
    most_recent_checkpoint = checkpoints[-1]
    return most_recent_checkpoint

def load_checkpoint_params_and_ctx(checkpoint_path):
    params = torch.load(checkpoint_path / "params.ckpt")
    ctx = torch.load(checkpoint_path / "context.ckpt")
    return params, ctx

@param("files.checkpoint_dir", "run.run_id")
def load_checkpoint(checkpoint_dir, run_id, trainer):
        print("Checkpoint dir: ", checkpoint_dir)

        checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.exists(), f"Checkpoint dir: {checkpoint_dir} does not exist"

        print(f"Loading state for run id: {run_id}")
        checkpoints = list((checkpoint_dir / run_id).glob("*"))

        def get_step(p):
            dir_name = str(p).split("/")[-1]
            return int(dir_name)

        checkpoints.sort(key=get_step)
        most_recent_checkpoint = checkpoints[-1]
        print(f"Resuming from checkpoint: {most_recent_checkpoint}")

        trainer.load(most_recent_checkpoint / "trainer.ckpt")
        params = torch.load(most_recent_checkpoint / "params.ckpt")
        ctx = torch.load(most_recent_checkpoint / "context.ckpt")
        return params, ctx

@param("files.checkpoint_dir", "checkpoint_dir")
def save_checkpoint(run_id, trainer, params, context, checkpoint_dir):
    steps = context["step"]
    path = Path(checkpoint_dir) / run_id / str(steps)
    path.mkdir(parents=True)

    trainer_ckpt = path / "trainer.ckpt"
    ctx_ckpt = path / "context.ckpt"
    params_ckpt = path / "params.ckpt"

    trainer.save(trainer_ckpt)
    torch.save(context, ctx_ckpt)
    torch.save(params, params_ckpt)

    if os.environ.get("WANDB_MODE") != "disabled":
        wandb.save(str(trainer_ckpt))
        wandb.save(str(ctx_ckpt))
        wandb.save(str(params_ckpt))

def create_trainer(params):
    unet1 = BaseUnet64(
        dim=params["dim"],
    )
    imagen = Imagen(
        unets=(unet1,),
        image_sizes=(64,),
        auto_normalize_img=True,
    ).cuda()
    return ImagenTrainer(imagen)

@param("run.run_id")
def run(run_id):
    run_id = run_id if run_id != "" else None
    params = {
        "dim": 256,
    }
    ctx = {
        "step": 0,
    }
    checkpoint_path  = get_checkpoint_path()
    assert bool(run_id) == bool(checkpoint_path)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        params, ctx = load_checkpoint_params_and_ctx(checkpoint_path)

    print("Params: ", params)
    print("Context: ", ctx)

    trainer = create_trainer(params)
    if checkpoint_path:
        trainer.load(checkpoint_path / "trainer.ckpt")

    resuming = checkpoint_path is not None
    run_id = run_id or wandb.util.generate_id()

    with wandb.init(
        project="Imagen",
        id=run_id,
        resume="must" if resuming else "never",
        config=params,
    ):
        try:
            train(run_id, trainer, params, ctx)
        except KeyboardInterrupt as e:
            print("Interrupted ...")
            save_checkpoint(run_id, trainer, params, ctx)

@param("data.batch_size", "batch_size")
@param("data.validations_per_epoch", "validations_per_epoch")
@param("logging.train_log_interval", "train_log_interval")
def train(run_id, trainer, params, ctx, batch_size, validations_per_epoch, train_log_interval):
    train_dl, val_dl = construct_dataloaders()
    trainer.train()

    val_interval = len(train_dl) // validations_per_epoch
    if val_dl:
        print("Validating every: ", val_interval)

    while True:
        epoch_step = 0
        for batch in tqdm(train_dl):
            ctx["step"] += 1
            epoch_step += 1

            imgs, tags = batch["img"], batch["tags"]
            loss = trainer(
                images=imgs,
                texts=tags,
                unet_number=1,
                max_batch_size=batch_size,
            )
            trainer.update(unet_number=1)

            if train_log_interval > 0 and epoch_step % train_log_interval == 0:
                wandb.log(
                    {
                        "loss": loss,
                        "step": ctx["step"],
                    }
                )

            if val_dl and epoch_step % val_interval == 0:
                print("Checkpointing ...")
                save_checkpoint(run_id, trainer, params, ctx)

                # Print non gc'd tensors
                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             print(type(obj), obj.size())
                #     except:
                #         pass

                trainer.eval()
                losses = []
                with torch.no_grad():
                    for batch in tqdm(val_dl):
                        imgs, tags = batch["img"].cuda(), batch["tags"]
                        loss = trainer.imagen(
                            images=imgs,
                            texts=tags,
                            unet_number=1,
                        )
                        losses.append(loss.item())
                wandb.log(
                    {
                        "loss_validation": sum(losses) / len(losses),
                        "step": ctx["step"],
                    }
                )
                trainer.train()

parser = argparse.ArgumentParser(description="Train a Imagen model")
config = get_current_config()
config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

run()