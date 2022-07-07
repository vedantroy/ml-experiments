import torch
from torch.optim import Adam
from torch import nn, optim
from tqdm import tqdm
from helpers import assert_shape


from params import params, constants
from data import AsciiDataset
from model import Transformer

model = Transformer(
    vocab_size=constants["vocab_size"],
    num_heads=params["heads"],
    d_model=params["d_model"],
    widening_factor=params["widening_factor"],
    sequence_len=params["sequence_len"],
    layers=params["layers"],
    # don't mask out anything
    mask=torch.ones((params["sequence_len"], params["sequence_len"])),
)

optimizer = Adam(params=model.parameters())
# TODO: Is there a way to print when the learning rate is being reduced?
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

# https://github.com/hyunwoongko/transformer/blob/master/train.py
def initialize_weights(m):
    if hasattr(m, "weight") and (m.weight is not None) and m.weight.dim() > 1:
        # Seems to be xavier initialization, but better?
        # https://pouannes.github.io/blog/initialization/
        nn.init.kaiming_uniform_(m.weight.data)


model.apply(initialize_weights)


def train_model(epochs: int):
    dataset = AsciiDataset(
        "./data/shakespeare.txt",
        batch_size=params["batch_size"],
        sequence_length=params["sequence_len"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        for batch_idx in tqdm(range(dataset.batches_per_epoch)):

            batch = next(dataset)
            x, y = batch["obs"], batch["target"]
            x = torch.from_numpy(x)

            optimizer.zero_grad()
            preds = model(x)
            assert_shape(
                preds,
                (params["batch_size"], params["sequence_len"], constants["vocab_size"]),
            )

            # blocked on: what should the end token be?


train_model(epochs=params["epochs"])
