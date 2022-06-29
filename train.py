import torch
from torch.optim import Adam
from torch import nn, optim
from tqdm import tqdm


from params import params, constants
from data import AsciiDataset
from model import Transformer

model = Transformer(
    vocab_size=constants["vocab_size"],
    num_heads=params["heads"],
    d_model=params["d_model"],
    widening_factor=params['widening_factor'],
    sequence_len=params['sequence_len'],
    layers=params['layers'],
    # don't mask out anything
    mask=torch.ones((params['sequence_len'], params['sequence_len']))
)

optimizer = Adam(params=model.parameters())
# TODO: Is there a way to print when the learning rate is being reduced?
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

# https://github.com/hyunwoongko/transformer/blob/master/train.py
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

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
            optimizer.zero_grad()
            model(batch['obs'])


train_model(epochs=params["epochs"])
