import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import UNet
from data import BasicDataset


def train_model(
    model,
    device,
    dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    percent_of_data_used_for_validation: float,
):
    n_train = int(len(dataset) * (percent_of_data_used_for_validation / 100))
    n_val = len(dataset) - int(n_train)
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
        pin_memory=False,
    )
    # shuffle=True causes the data to be reshuffled at every epoch
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            imgs, masks = batch["image"], batch["mask"]
            actual_batch_size, channels, W, H = imgs.shape
            assert actual_batch_size == batch_size
            assert masks.shape == (actual_batch_size, W, H)

            preds = model(imgs)
            _ ,_, predW, predH = preds.shape
            assert predW == W and predH == H
            print("1 done!")


# https://github.com/hyunwoongko/transformer/blob/master/train.py
def initialize_weights(m):
    if hasattr(m, "weight") and (m.weight is not None) and m.weight.dim() > 1:
        # Seems to be xavier initialization, but better?
        # https://pouannes.github.io/blog/initialization/
        nn.init.kaiming_uniform_(m.weight.data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = BasicDataset(
        images_dir="./data/train", masks_dir="./data/train_masks", scale=0.5
    )
    logging.info(f"{len(dataset)} training images")

    model = UNet(n_in_channels=3, n_classes=2)
    # TODO: Do check-pointing
    model.apply(initialize_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    logging.info(f"Training on device: {device}")

    train_model(
        model=model,
        device=device,
        dataset=dataset,
        epochs=5,
        batch_size=1,
        learning_rate=1e-5,
        percent_of_data_used_for_validation=10,
    )
