import logging
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
import random
from tqdm import tqdm

from model import UNet
from data import BasicDataset
from loss import dice_loss

def train_model(
    model,
    device,
    dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    percent_of_data_used_for_validation: float,
):
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
        pin_memory=False,
    )
    # shuffle=True causes the data to be reshuffled at every epoch
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # adamW > adam -- https://www.fast.ai/2018/07/02/adam-weight-decay/#implementing-adamw
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # The dice-coefficient (how much 2 sets intersect)
    # is good for forcing the segmentation to be sharp b/c it
    # only cares about the relative overlap between the prediction & ground truth
    # But this is has some variance issues, so we augment it with cross-entropy loss
    # See:
    # https://stats.stackexchange.com/questions/438494/what-is-the-intuition-behind-what-makes-dice-coefficient-handle-imbalanced-data
    # https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            imgs, masks = batch["image"], batch["mask"]
            actual_batch_size, channels, W, H = imgs.shape
            assert actual_batch_size == batch_size
            assert masks.shape == (actual_batch_size, W, H)

            imgs = imgs.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)

            # `set_to_none=True` boosts performance
            optimizer.zero_grad(set_to_none=True)
            masks_pred = model(imgs)

            _ , out_classes, predW, predH = masks_pred.shape
            assert predW == W and predH == H and out_classes == model.n_classes

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
            ground_truth = F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2).float()

            # this is only true if N_CLASSES=2
            x, y = random.randrange(0, W), random.randrange(0, H)
            assert 1. - (probs[0][0][x][y] + probs[0][1][x][y]).item() < 1e-6

            loss = criterion(masks_pred, masks) + dice_loss(probs, ground_truth)
            loss.backward()
            optimizer.step()

# https://github.com/hyunwoongko/transformer/blob/master/train.py
def initialize_weights(m):
    if hasattr(m, "weight") and (m.weight is not None) and m.weight.dim() > 1:
        # Seems to be xavier initialization, but better?
        # https://pouannes.github.io/blog/initialization/
        nn.init.kaiming_uniform_(m.weight.data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = BasicDataset(
        images_dir="./data/imgs", masks_dir="./data/masks", scale=0.5
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
