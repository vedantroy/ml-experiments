from torch import nn
import torch.nn.functional as F
import torch as th

# OpenAI's diffusion model defaults
# NOTE: some of these are NOT optimal--e.g., using linear instead of cosine schedule
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
# def model_and_diffusion_defaults():
#     """
#     Defaults for image training.
#     """
#     return dict(
#         image_size=64,
#         num_channels=128,
#         num_res_blocks=2,
#         num_heads=4,
#         num_heads_upsample=-1,
#         attention_resolutions="16,8",
#         dropout=0.0,
#         learn_sigma=False,
#         sigma_small=False,
#         class_cond=False,
#         diffusion_steps=1000,
#         noise_schedule="linear",
#         timestep_respacing="",
#         use_kl=False,
#         predict_xstart=False,
#         rescale_timesteps=True,
#         rescale_learned_sigmas=True,
#         use_checkpoint=False,
#         use_scale_shift_norm=True,
#     )

# The actual parameters that get passed into OpenAI's model
# NOTE: Unclear if these are optimal
# model_channels=128
# out_channels=3
# num_res_blocks=3
# attention_resolutions=(4, 8)
# dropout=0.0
# channel_mult=(1, 2, 3, 4)
# conv_resample=True
# dims=2
# num_classes=None
# use_checkpoint=False
# num_heads=4
# use_scale_shift_norm=True

# Implementation of Open AI's improved diffusion
# https://arxiv.org/abs/2102.09672

# QUESTIONS from OpenAI implementation:
# 1. What is the purpose of 1D / 3D convolutions?
# 2. Why is GroupNorm done in float32?
# 3. Why add 1 to scale shift? (My guess is that it prevents gradient issues?)
# - DONE
# 4. What is the purpose of `zero_module`?
#   - Seems like it initializes the weights to 0; but isn't that bad?
# 5. (for Abhi) Why do we use a set dimension for the positional encodings & then pass them through a MLP?
#   I kind of understand the MLP part, but why not just make the PEs equal to the channels?

# CONVENTIONS:
# - All non-learned parameters are applied using F.func in the `forward` method
# - This makes it clear what is learned

# TODO: QUESTION(2)
def normalization(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MyResBlock(nn.Module):
    """
    This looks like the combination of 2 architectures:
    - UNet layer 2 x (3x3 conv + ReLU): https://arxiv.org/pdf/1505.04597.pdf
    - Residual / skip connections:  https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_channels = time_emb_channels

        self.in_norm = normalization(in_channels)
        self.in_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        # TODO: QUESTION(5)
        self.time_emb_linear = nn.Linear(
            in_features=time_emb_channels, out_features=2 * out_channels
        )

        self.out_norm = normalization(out_channels)

        # > 1. We zero the skip connections following Ho et al 2020). 
        # > Like you note, this does initialize resblocks to the identity, 
        # > which can actually help stabilize training (https://arxiv.org/abs/1901.09321). 
        # > This doesn't actually prevent learning. In the first step, the gradient for 
        # > most of the resblock is indeed zero, but in subsequent steps it will not be zero 
        # > because the zero'd out weight will itself no longer be zero.
        self.out_conv = zero_module(nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        ))

        # OpenAI implementation offers choice between convolution for skip connections &
        # up or down sample, but lucidrains just uses 1x1 convolution
        # No mention of improved performance in paper, so skip for simplicity
        if in_channels == out_channels:
            self.skip_projection = nn.Identity()
        else:
            self.skip_projection = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )

    def forward(self, x, emb, dbg=None):
        if dbg == None:
            dbg = {}
        N, in_C, H, W = x.shape
        assert in_C == self.in_channels

        _x = x
        x = self.in_norm(x)
        x = F.silu(x)
        x = self.in_conv(x)
        assert x.shape == (N, self.out_channels, H, W)

        # > Additionally, we changed the
        # > way the model conditions on t. In particular, instead of
        # > computing a conditioning vector v and injecting it into hidden
        # > state h as GroupNorm(h + v), we compute conditioning
        # > vectors w and b and inject them into the hidden state as
        # > GroupNorm(h)(w + 1) + b.

        # Explanation: They convert the embedding into weight + bias
        # and skip passing it through group normalization
        dbg["emb_in"] = emb
        emb = F.silu(emb)
        assert emb.shape == (N, self.time_emb_channels)
        emb = self.time_emb_linear(emb)
        dbg["emb"] = emb
        assert emb.shape == (N, self.out_channels * 2)
        cond_w, cond_b = th.chunk(emb[..., None, None], 2, dim=1)
        dbg["cond_w"], dbg["cond_b"] = cond_w, cond_b
        assert cond_w.shape == cond_b.shape
        assert cond_w.shape == (N, self.out_channels, 1, 1)

        # > 2. We add one to the scale parameter so that the scale is centered around 1. 
        # > This helps preserve the stddev of activations flowing through the model--important 
        # > for training stability.

        # My understanding: Think back to ResNet. If the optimal scale 
        # (where scale == `1 + cond_w`) is the identity (e.g scale == 1), then
        # `cond_w` will be optimized to 0. Generally in ML, you want data to be
        # centered around 0 and normalized by the stdev, so this reminds of that
        # NOTE: This could be totally wrong
        x = self.out_norm(x) * (1 + cond_w) + cond_b
        dbg["scale_shift_norm"] = x
        x = F.silu(x)
        x = self.out_conv(x)

        # The OpenAI implementation has dropout right here, but also says
        #   > We then tried runs with dropout 0.1 and 0.3, and
        #   > found that models with a small amount of dropout improved
        #   > the best attainable FID but took longer to get to the same
        #   > performance and still eventually overfit. We concluded that
        #   > the best way to train, given what we know, is to early stop
        #   > and instead increase model size if we want to use add
        # so, skip dropout
        return self.skip_projection(_x) + x


class UNet(nn.Module):
    def __init__(self):
        # > The downsampling stack performs four steps of
        # > downsampling, each with three residual blocks
        pass
