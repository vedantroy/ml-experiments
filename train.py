import torch as th
import yahp as hp
from composer import ComposerModel, Trainer, TimeUnit
from composer.datasets.streaming import StreamingDataset
from diffusion.diffusion import GaussianDiffusion, cosine_betas
from unet.unet import UNet
from torch.utils.data import DataLoader

from types import SimpleNamespace
from dataclasses import dataclass
from typing import List

from utils import load_tensor


@dataclass
class UNetParams(hp.Hparams):
    in_channels: int = hp.required("# input channels")
    out_channels: int = hp.required("# output channels")
    # (C in [0] under Appendix A "Hyperparameters")
    model_channels: int = hp.required("# model channels")
    channel_mult: List[int] = hp.required("the channel multipliers")
    layer_attn: List[bool] = hp.required(
        "whether to use attention between ResNet blocks"
    )
    res_blocks: int = hp.required("# ResNet blocks")
    attention_heads: int = hp.required("# attention heads")

    def initialize_object(self):
        return UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            layer_attn=self.layer_attn,
            num_res_blocks=self.res_blocks,
            num_heads=self.attention_heads,
        )


@dataclass
class DiffusionParams(hp.Hparams):
    steps: int = hp.required("# diffusion steps")
    schedule: str = hp.required("diffusion schedule")

    def initialize_object(self):
        assert self.schedule == "cosine", "Only cosine schedule is supported"
        betas = cosine_betas(self.steps)
        return GaussianDiffusion(betas)


@dataclass
class TrainerConfig(hp.Hparams):
    unet: UNetParams = hp.required("the UNet model")
    diffusion: DiffusionParams = hp.required("Gaussian diffusion parameters")

    def initialize_object(self):
        return self.unet.initialize_object(), self.diffusion.initialize_object()


class IDDPM(ComposerModel):
    def __init__(self, unet: UNet, diffusion: GaussianDiffusion):
        super().__init__()
        self.model = unet
        self.diffusion = diffusion

    def forward(self, batch):
        batch = batch["img"]
        N, *_ = batch.shape
        # normalize images to [-1, 1]
        batch = ((batch / 255) * 2) - 1

        print(N)

        # Only support uniform sampling
        t = th.randint(self.diffusion.n_timesteps, (N,))

        x_0 = batch
        noise = th.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)
        model_out = self.model(batch, t)
        d = dict(x_0=x_0, x_t=x_t, noise=noise, model_out=model_out)
        return SimpleNamespace(**d)

    def loss(self, out, _):
        mse_loss, vb_loss = self.diffusion.training_losses(
            out.x_0, out.x_t, out.t, out.noise
        )
        return mse_loss + vb_loss


if __name__ == "__main__":
    config = TrainerConfig.create("./config/basic.yaml", None, cli_args=False)
    unet, diffusion = config.initialize_object()
    iddpm = IDDPM(unet, diffusion)

    ds = StreamingDataset(
        remote=None,
        local="./dataset",
        decoders={"img": load_tensor},
        batch_size=4,
        shuffle=True,
    )
    train_dl = DataLoader(ds, batch_size=None, shuffle=False)

    trainer = Trainer(
        model=iddpm,
        train_dataloader=train_dl,
        eval_dataloader=None,
        schedulers=[],
        optimizers=[],
        max_duration="10ep",
        device="gpu",
        precision="amp",
    )
    trainer.fit()