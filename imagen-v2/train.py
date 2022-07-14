import os
import argparse
from pathlib import Path
import random

from tqdm import tqdm
from imagen_pytorch import Imagen, BaseUnet64, ImagenTrainer
from torch.utils.data import DataLoader, random_split
import torch
import wandb

from streaming_dataset import ImageWithCaptions

config = {
    "dim": 256,
    "batch_size": 1,
}

batch_size = config["batch_size"]
ds = ImageWithCaptions("./data/danbooru/artifacts/shards_very_small_3", shuffle=True, batch_size=batch_size)

train_sz = int(len(ds) * 0.05)
val_sz = len(ds) - train_sz

train_ds, val_ds = random_split(ds, [train_sz, val_sz])
print(f"train: {len(train_ds)}, val: {len(val_ds)}")
# print(f"train: {len(train_ds)}")

train_dl = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True,
    num_workers=12,
)
# val_dl = DataLoader(val_ds, batch_size, shuffle=False, pin_memory=True, num_workers=4)

def test_dataloader():
    print("Testing dataloader ...")
    i = 0
    for batch in tqdm(train_dl):
        i += 1

# Uncomment this if you want to see how long it takes to just run the dataloader
# test_dataloader()

def save(id, trainer, metadata):
    steps = metadata["step"]
    path = Path(f"./runs/{id}") / str(steps)
    path.mkdir(exist_ok=False, parents=True)

    trainer_ckpt = path / "trainer.ckpt"
    metadata_ckpt = path / "metadata.ckpt"

    trainer.save(trainer_ckpt)
    torch.save(metadata, metadata_ckpt)

    if os.environ.get("WANDB_MODE") != "disabled":
        wandb.save(str(trainer_ckpt))
        wandb.save(str(metadata_ckpt))


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

# text => img UNet
unet1 = BaseUnet64(
    # do 1/2 the dim (speedup training size)
    dim=config["dim"],
)

imagen = Imagen(
    unets=(unet1,),
    image_sizes=(64,),
    # If True, this just does 0 to 1 => -1 to 1
    auto_normalize_img=True,
).cuda()
trainer = ImagenTrainer(imagen)
trainer.train()

# By batching wandb syncing, we 2x training speed
def commit():
    should_commit = random.randint(0, 100) == 0
    return should_commit

def run():
    step = 0

    if resuming:
        print(f"Loading state for run id: {run_id}")
        checkpoints = list(Path(f"./runs/{run_id}").glob("*"))

        def get_step(p):
            dir_name = str(p).split("/")[-1]
            return int(dir_name)

        checkpoints.sort(key=get_step)
        most_recent_checkpoint = checkpoints[-1]
        print(f"Resuming from checkpoint: {most_recent_checkpoint}")

        trainer.load(most_recent_checkpoint / "trainer.ckpt")
        metadata = torch.load(most_recent_checkpoint / "metadata.ckpt")
        step = metadata["step"]

    LOG_SCALING_FACTOR = 0.1
    while True:
        for batch in tqdm(train_dl):
            step += 1
            imgs, tags = batch["img"], batch["tags"]

            loss = trainer(
                images=imgs,
                texts=tags,
                unet_number=1,
                max_batch_size=batch_size,
            )
            trainer.update(unet_number=1)

            if step % (50 * LOG_SCALING_FACTOR) == 0:
                # Logging every step is too slow
                wandb.log(
                    {
                        "loss": loss,
                        "step": step,
                    })


            # checkpoint model & do sample run
            if step % 5000 * LOG_SCALING_FACTOR == 0 and False:
                print("Checkpointing ...")
                save(run_id, trainer, {"step": step})

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
                            unet_number = 1,
                        )
                        losses.append(loss.item())
                wandb.log(
                    {
                        "loss_validation": sum(losses) / len(losses),
                        "step": step,
                    }
                    #}, commit=commit()
                )
                trainer.train()

                # sample runs take a *long* time (~ 3 minutes)
                # if step % 20 == 0:
                #     # sample run time
                #     N_SAMPLES = 3
                #     imgs = trainer.sample(
                #         sample_batch["tags"][:N_SAMPLES], cond_scale=3.0
                #     )

                #     trips = [
                #         (
                #             ("prompt" + str(idx), sample_batch["tags"][idx]),
                #             ("img" + str(idx), sample_batch["img"][idx]),
                #             ("actual" + str(idx), imgs[idx]),
                #         )
                #         for idx in range(N_SAMPLES)
                #     ]

                #     to_log = {}
                #     for pairs in trips:
                #         for (k, v) in pairs:
                #             to_log[k] = v
                #     wandb.log(to_log)

with wandb.init(
    project="Imagen", id=run_id, resume="must" if resuming else "never", config=config,
):
    try:
        run()
    except KeyboardInterrupt as e:
        print("Interrupted ...")
        save(run_id, trainer, {"step": step})
