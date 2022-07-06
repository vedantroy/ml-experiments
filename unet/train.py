from pathlib import Path
import argparse
import logging
# import contextlib
import random

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from validation import evaluate
from model import UNet
from data import BasicDataset
from loss import dice_loss
from utils import convert_mask_to_ground_truth

# These need to be global so we can save them
BATCHES_SEEN = EPOCH = MODEL = OPTIMIZER = GRAD_SCALER = SAMPLE_ARG = None

def train_model(
    epochs,
    batch_size,
    learning_rate,
    percent_of_data_used_for_validation,
    scale,
    amp,
    checkpoint,
):
    global BATCHES_SEEN, EPOCH, MODEL, OPTIMIZER, GRAD_SCALER, SAMPLE_ARG

    dataset = BasicDataset(
        images_dir="./data/imgs", masks_dir="./data/masks", scale=scale
    )
    logging.info(f"{len(dataset)} training images")

    MODEL = UNet(n_in_channels=3, n_classes=2)
    if checkpoint:
        MODEL.load_state_dict(checkpoint["model_state_dict"])
    else:
        # TODO: Do check-pointing
        MODEL.apply(initialize_weights)
    MODEL.train()

    using_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(using_cuda)
    MODEL.to(device=device)
    logging.info(f"Training on device: {device}")

    n_val = int(len(dataset) * (percent_of_data_used_for_validation / 100))
    n_train = len(dataset) - int(n_val)
    assert n_train + n_val == len(dataset)

    # The dataset is not an iterator so we can't just use slice syntax to split it
    # `random_split` splits a dataset into 2 new datasets; it also shuffles the dataset before splitting (I think)
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # The Dataset retrieves our dataset’s features and labels one sample at a time.
    # While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting,
    # and use Python’s multiprocessing to speed up data retrieval.
    # DataLoader handles that

    loader_args = dict(
        batch_size=batch_size,
        num_workers=4,
        # Only useful if you're training on GPU
        pin_memory=using_cuda,
    )
    # shuffle=True causes the data to be reshuffled at every epoch
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # `drop_last=True` ignores the last batch (if the number of samples is not divisible by the batch size)
    # i.e., it drops the "ragged" data
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    assert (
        len(val_loader) > 0
    ), "validation loader didn't have enough data for a single full batch"

    # adamW > adam -- https://www.fast.ai/2018/07/02/adam-weight-decay/#implementing-adamw
    OPTIMIZER = optim.AdamW(MODEL.parameters(), lr=learning_rate)
    if checkpoint:
        OPTIMIZER.load_state_dict(checkpoint["optimizer_state_dict"])

    GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=amp)
    if checkpoint:
        GRAD_SCALER.load_state_dict(checkpoint["grad_scaler_state_dict"])

    # The dice-coefficient (how much 2 sets intersect)
    # is good for forcing the segmentation to be sharp b/c it
    # only cares about the relative overlap between the prediction & ground truth
    # But this is has some variance issues, so we augment it with cross-entropy loss
    # See:
    # https://stats.stackexchange.com/questions/438494/what-is-the-intuition-behind-what-makes-dice-coefficient-handle-imbalanced-data
    # https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b

    criterion = nn.CrossEntropyLoss()

    batches_per_epoch = len(train_loader)
    # If your training gets much slower, increase `log_freq`
    wandb.watch(MODEL, criterion=None, log="all", log_freq=10)
    BATCHES_SEEN = 0 if not checkpoint else checkpoint["batches_seen"]
    start_epoch = 0 if not checkpoint else checkpoint["epoch"]
    batches_per_epoch = len(train_loader)
    batches_to_skip = 0
    if checkpoint:
        batches_to_skip = BATCHES_SEEN - (start_epoch * batches_per_epoch)

    MODEL.train()
    for epoch in range(start_epoch, epochs):
        EPOCH = epoch
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")):
            if batches_to_skip > 0:
                # print(f"Skipping {batches_to_skip} more batches")
                batches_to_skip -= 1
                continue

            imgs, masks = batch["image"], batch["mask"]
            SAMPLE_ARG = imgs
            actual_batch_size, channels, W, H = imgs.shape
            assert actual_batch_size == batch_size
            assert masks.shape == (actual_batch_size, W, H)

            imgs = imgs.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)

            # `set_to_none=True` boosts performance
            OPTIMIZER.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = MODEL(imgs)

                _, out_classes, predW, predH = masks_pred.shape
                assert predW == W and predH == H and out_classes == MODEL.n_classes

                # Imagine if the mask is:
                # 0 0 1 0 0
                # So, the center pixel is the car/whatever
                # `ground_truth.permute(...)` will look like:
                # 1 1 0 1 1 <= probability pixel is 1st class
                # 0 0 1 0 0 <= probability pixel is 2nd class
                # and `probs` might look like:
                # 0.9 0.9 0.2 0.8 0.7
                # 0.1 0.1 0.8 0.2 0.3

                probs = F.softmax(masks_pred, dim=1).float()
                ground_truth = convert_mask_to_ground_truth(masks, MODEL.n_classes)

                # this is only true if N_CLASSES=2
                # x, y = random.randrange(0, W), random.randrange(0, H)
                # assert 1.0 - (probs[0][0][x][y] + probs[0][1][x][y]).item() < 1e-6

                loss_cross_entropy = criterion(masks_pred, masks)
                loss_dice = dice_loss(probs, ground_truth)
                loss = loss_cross_entropy + loss_dice

            GRAD_SCALER.scale(loss).backward()
            GRAD_SCALER.step(OPTIMIZER)
            GRAD_SCALER.update()

            BATCHES_SEEN += 1

            to_log = {
                "loss_combined": loss.item(),
                "loss_cross_entropy": loss_cross_entropy,
                "loss_dice": loss_dice,
                "step": BATCHES_SEEN,
                "epoch": epoch,
                "learning_rate": OPTIMIZER.param_groups[0]["lr"],
            }

            if BATCHES_SEEN % 100 == 1:
                wandb.log(
                    {
                        **to_log,
                        "image": wandb.Image(imgs[0].cpu()),
                        "masks": {
                            "true": wandb.Image(masks[0].float().cpu()),
                            "pred": wandb.Image(
                                # The original repository has a softmax, but
                                # that should be unneeded since softmax never changes argmax
                                # TODO: Check if we can do softmax(masks_pred[0], dim=0)
                                # torch.softmax(masks_pred, dim=1)
                                masks_pred.argmax(dim=1)[0]
                                .float()
                                .cpu()
                            ),
                        },
                    }
                )
            else:
                wandb.log(to_log)

            if (batch_idx + 1) % (batches_per_epoch // 2) == 0:
                val_loss_dice, val_loss_cross_entropy, val_loss_total, = evaluate(MODEL, val_loader, device)
                wandb.log({
                    "step": BATCHES_SEEN,
                    "validation_loss_dice": val_loss_dice,
                    "validation_loss_cross_entropy": val_loss_cross_entropy,
                    "validation_loss_combined": val_loss_total,
                })


# https://github.com/hyunwoongko/transformer/blob/master/train.py
def initialize_weights(m):
    if hasattr(m, "weight") and (m.weight is not None) and m.weight.dim() > 1:
        # Seems to be xavier initialization, but better?
        # https://pouannes.github.io/blog/initialization/
        nn.init.kaiming_uniform_(m.weight.data)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--load",
        "-L",
        type=str,
        default=False,
        help="Load model from a checkpoint directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    run_id = checkpoint = None
    if args.load:
        run_id = args.load
        path = f"./run-checkpoints/{run_id}.checkpoint"
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Could not find file at: {path}")
        checkpoint = torch.load(p)
    else:
        run_id = wandb.util.generate_id()
        logging.info(f"Starting run with id: {run_id}")

    hyperparameters = (
        dict(
            epochs=5,
            # A NVIDIA Quadro 4000 8GB can handle a batch size of 2
            # but it seems to be slower (30min per epoch vs 26)
            batch_size=1,
            percent_of_data_used_for_validation=10,
            # how to scale the images by before passing them to the model
            # kind of a hyperparam?
            scale=0.5,
            # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
            # Folk wisdom for adam's learning rate
            # TODO: do hyper-parameter tuning or something?
            learning_rate=3e-4,
            amp=False,
        )
        if not checkpoint
        else checkpoint["hyperparameters"]
    )

    throw = None
    # as long as the process exits unsuccesfully, wandb will automatically resume the run
    with wandb.init(
        project="UNet",
        id=run_id,
        resume="must" if checkpoint else "never",
        config=hyperparameters,
    ):
        config = wandb.config
        try:
            train_model(**config, checkpoint=checkpoint)
        except KeyboardInterrupt as e:
            throw = e

    path = Path(f"./run-checkpoints")
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batches_seen": BATCHES_SEEN,
            "epoch": EPOCH,
            "model_state_dict": MODEL.state_dict(),
            "optimizer_state_dict": OPTIMIZER.state_dict(),
            "grad_scaler_state_dict": GRAD_SCALER.state_dict(),
            "hyperparameters": hyperparameters,
        },
        path / f"{run_id}.checkpoint",
    )
    logging.info("Saved interrupt")

    if throw:
        raise throw
    else:
        model_path = path / f"{run_id}.pt"
        model_onnx_path = path / f"{run_id}.onnx"
        torch.save(MODEL.state_dict(), model_path)
        torch.onnx.export(MODEL, SAMPLE_ARG, model_onnx_path)
        wandb.save(model_path)
        wandb.save(model_onnx_path)


