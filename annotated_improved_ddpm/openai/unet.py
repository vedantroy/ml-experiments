from abc import abstractmethod

import math
from sre_constants import ASSERT

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


def print2(*args):
    if False:
        print(*args)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        print2("======")
        print2(f"dims: {dims}")
        self.channels = channels
        print2(f"channels: {channels}")
        self.emb_channels = emb_channels
        print2(f"emb_channels: {emb_channels}")
        self.dropout = dropout
        print2(f"dropout: {dropout}")
        self.out_channels = out_channels or channels
        print2(f"out_channels: {out_channels}")
        self.use_conv = use_conv
        print2(f"use_conv: {use_conv}")
        self.use_checkpoint = use_checkpoint
        print2(f"use_checkpoint: {use_checkpoint}")
        self.use_scale_shift_norm = use_scale_shift_norm
        print2(f"use_scale_shift_norm: {use_scale_shift_norm}")

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, dbg=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if dbg == None:
            dbg = {}
        return checkpoint(
            self._forward, (x, emb, dbg), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, dbg):
        N, C, H, W = x.shape
        assert emb.shape == (N, self.emb_channels)

        h = self.in_layers(x)
        assert h.shape == (N, self.out_channels, H, W)
        dbg["emb_in"] = emb
        emb_out = self.emb_layers(emb).type(h.dtype)
        dbg["emb"] = emb_out
        # Now I see why this is a loop
        # While the repository does diffusion on 2D data (images)
        # It could be done on higher dimensional data
        # (so maybe x.shape == (N, C, W, H, ?))
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        assert emb_out.shape == (N, 2 * self.out_channels, 1, 1)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            assert scale.shape == shift.shape
            assert scale.shape == (N, self.out_channels, 1, 1)
            dbg["cond_w"], dbg["cond_b"] = scale, shift
            h = out_norm(h) * (1 + scale) + shift
            assert h.shape == (N, self.out_channels, H, W)
            dbg["scale_shift_norm"] = h
            # SilU + Dropout + conv_nd
            assert len(out_rest) == 3
            h = out_rest(h)
            assert h.shape == (N, self.out_channels, H, W)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        assert self.qkv.weight.data.shape == (channels * 3, channels, 1)
        self.attention = QKVAttention()
        # QUESTION: Why is this final convolution needed?
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        H, W = spatial
        # images are square
        assert H == W
        # Flatten the input into a single sequence
        x = x.reshape(b, c, -1)
        assert x.shape == (b, c, H * W)
        _x = self.norm(x)
        assert x.shape == (b, c, H * W)
        qkv = self.qkv(_x)
        assert qkv.shape == (b, c * 3, H * W)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        assert qkv.shape == (b * self.num_heads, c * 3 / self.num_heads, H * W)
        h = self.attention(qkv)
        # the 2nd dimension represents the re-averaged value vectors
        # `c / self.num_heads` = the length of a value vector
        assert h.shape == (b * self.num_heads, c / self.num_heads, H * W)
        h = h.reshape(b, -1, h.shape[-1])
        # concatenate the value vectors
        assert h.shape == (b, c, H * W)
        h = self.proj_out(h)
        assert h.shape == (b, c, H * W)
        # reshape back into H, W
        # Also, this is a residual connection inside attention?
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        N, seq_len = qkv.shape[0], qkv.shape[2]
        # There's some odd thing in the docs, (maybe I was on an outdated version)
        # where it says the 2nd param = the # of splits (when it's actually the size of a split)
        q, k, v = th.split(qkv, ch, dim=1)
        assert q.shape == (N, ch, seq_len)
        assert q.shape == k.shape and k.shape == v.shape
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        assert weight.shape == (N, seq_len, seq_len)
        r = th.einsum("bts,bcs->bct", weight, v)
        assert r.shape == (N, ch, seq_len)
        return r

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            # print("adding res blocks")
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # I'm not sure why they're not specifying attention by using the level
                # Specifically, I don't really get the point of the ds parameter
                print2(f"level {level}, ds={ds} attn={ds in attention_resolutions}")
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                # print("adding downsample")
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
                # print(ds)

        # print(input_block_chans)
        # print(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        # `+ input_block_chans.pop()` => represents the skip connection
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def print_architecture(self):
        input_blocks = self.input_blocks
        middle_block = self.middle_block
        output_blocks = self.output_blocks

        def print_timestep(seq):
            for l in seq:
                if isinstance(l, ResBlock):
                    print(
                        f"ResBlock(in={l.channels}, out={l.out_channels}, emb_channels={l.emb_channels})"
                    )
                elif isinstance(l, AttentionBlock):
                    print(f"AttentionBlock(in={l.channels}, heads={l.num_heads})")
                elif isinstance(l, nn.Conv2d):
                    print(f"Conv2d(in={l.in_channels}, out={l.out_channels})")
                elif isinstance(l, Downsample):
                    print(f"Downsample(in={l.channels})\n")
                elif isinstance(l, Upsample):
                    print(f"Upsample(in={l.channels})\n")
                else:
                    print("Unknown layer ...")

        def print_blocks(x):
            if isinstance(x, TimestepEmbedSequential):
                print_timestep(x)
            elif isinstance(x, nn.Sequential):
                for l in x:
                    if isinstance(l, nn.GroupNorm) or isinstance(l, SiLU):
                        continue
                    elif isinstance(l, nn.Conv2d):
                        print(f"Conv2d(in={l.in_channels}, out={l.out_channels})")
                    else:
                        print("Unknown layer ...")
            else:
                for seq in x:
                    assert isinstance(seq, TimestepEmbedSequential)
                    print_timestep(seq)

        print("INPUT BLOCKS:")
        print_blocks(input_blocks)
        print("\n\nMIDDLE BLOCKS:")
        print_blocks(middle_block)
        print("\n\nOUTPUT BLOCKS:")
        print_blocks(output_blocks)
        print("\n\nOUT:")
        print_blocks(self.out)
        print("\n\n")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        N, C, W, H = x.shape
        assert timesteps.shape == (N,)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        def print_dims(x, skip=None):
            _, _, H, W = x.shape
            if skip == None:
                print(f"{tuple(x.shape)} -> {H}x{W}")
            else:
                print(f"{tuple(x.shape)} -> {H}x{W} (skip = {tuple(skip.shape)})")

        h = x.type(self.inner_dtype)
        print("INPUT:")
        print_dims(h)
        print("INNER:")
        for module in self.input_blocks:
            h = module(h, emb)
            print_dims(h)
            hs.append(h)
        h = self.middle_block(h, emb)
        print("MIDDLE:")
        print_dims(h)
        print("OUTER:")
        for module in self.output_blocks:
            skip = hs.pop()
            cat_in = th.cat([h, skip], dim=1)
            h = module(cat_in, emb)
            print_dims(h, skip=skip)
        h = h.type(x.dtype)
        r = self.out(h)
        print("OUTPUT:")
        print_dims(r)
        return r

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
