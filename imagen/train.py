import argparse
from pathlib import Path
from random import random

from tqdm import tqdm
from imagen_pytorch import Imagen, BaseUnet64, ImagenTrainer
from torch.utils.data import DataLoader, random_split
import torch
import wandb

from dataloader import ImageCaptionDataset

# @ batch_size = 2 => 1.71it/s = 3.42
# @ batch_size = 4 => 1.08it/s = 4.32
# crashes @ batch_size = 6


def save(id, trainer, metadata):
    steps = metadata["steps"]
    path = Path(f"./runs/{id}") / str(steps)
    path.mkdir(exist_ok=False, parents=True)

    trainer_ckpt = path / "trainer.ckpt"
    metadata_ckpt = path / "metadata.ckpt"

    trainer.save(trainer_ckpt)
    torch.save(metadata, metadata_ckpt)

    wandb.save(trainer_ckpt)
    wandb.save(metadata_ckpt)


def get_args(description):
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--id",
        "-I",
        type=str,
        default=False,
        help="Load model using an existing run id",
    )
    return parser.parse_args()


args = get_args("Train Imagen")

# mutable global state :)

resuming = bool(args.id)
run_id = args.id if resuming else wandb.util.generate_id()
step = 0

config = {
    "dim": 256,
    "batch_size": 4,
}

# text => img UNet
unet1 = BaseUnet64(
    # do 1/2 the dim (speedup training size)
    dim=config["dim"],
)

imagen = Imagen(
    unets=(unet1,),
    image_sizes=(64,),
    auto_normalize_img=False,
).cuda()

ds = ImageCaptionDataset(
    img_dir_path="./data/danbooru/data/dataset/imgs",
    caption_server_path="./data/danbooru/scripts/query_server.js",
)

batch_size = config["batch_size"]
train_ds, val_ds = random_split(ds, [len(ds) - batch_size * 5, batch_size * 5])
print(f"# images: {len(train_ds)}")

train_dl = DataLoader(
    train_ds, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
val_dl = DataLoader(val_ds, batch_size, shuffle=False, pin_memory=True)

train_dl_iter = iter(train_dl)
trainer = ImagenTrainer(imagen)


def run():
    global step

    if run_id:
        checkpoints = list(Path(f"./runs/{run_id}").glob("*"))

        def get_step(p):
            dir_name = p.split("/")[-1]
            return int(dir_name)

        checkpoints.sort(key=get_step)
        most_recent_checkpoint = checkpoints[-1]

        trainer.load(most_recent_checkpoint / "trainer.ckpt")
        metadata = torch.load(most_recent_checkpoint / "metadata.ckpt")
        step = metadata["step"]

    while True:
        for batch in tqdm(train_dl_iter):
            batch_size = config["batch_size"]
            assert batch["img"].shape[0] == batch_size
            assert len(batch["caption"]) == batch_size

            loss = trainer(
                images=batch["img"],
                texts=batch["caption"],
                unet_number=1,
                max_batch_size=batch_size,
            )

            wandb.log(
                {
                    "loss": loss,
                    "step": step,
                }
            )

            step += 1

            # checkpoint model & do sample run
            if step % 1000 == 0:
                print("Checkpointing ...")
                save(run_id, trainer, {"step": step})

                last_batch = None
                losses = []
                for batch in val_dl:
                    last_batch = batch
                    with torch.no_grad():
                        loss = trainer(
                            images=batch["img"],
                            text=batch["caption"],
                            unet_number=1,
                            max_batch_size=batch_size,
                        )
                        losses.append(loss)

                wandb.log(
                    {
                        "loss_validation": sum(losses) / len(losses),
                        "step": step,
                    }
                )

                # sample run time
                N_SAMPLES = 3
                imgs = trainer.sample(last_batch["caption"][:N_SAMPLES], cond_scale=3.0)

                trips = [
                    (
                        ("prompt" + str(idx), last_batch["caption"][idx]),
                        ("img" + str(idx), last_batch["img"][idx]),
                        ("actual" + str(idx), imgs[idx]),
                    )
                    for idx in range(N_SAMPLES)
                ]

                to_log = {}
                for pairs in trips:
                    for (k, v) in pairs:
                        to_log[k] = v
                wandb.log(to_log)

with wandb.init(
    project="Imagen", id=run_id, resume="must" if resuming else "never", config=config
):
    try:
        run()
    except KeyboardInterrupt:
        print("Interrupted ...")
        save(run_id, trainer, {"steps": step})
