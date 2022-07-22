import torch as tch

from unet import UNetModel

# These are parameters I copied by doing `print(...)`
# in the original repo
model = UNetModel(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=3,
    attention_resolutions=(4, 8),
    dropout=0.0,
    channel_mult=(1, 2, 3, 4),
    conv_resample=True,
    num_classes=None,
    use_checkpoint=False,
    num_heads=4,
    num_heads_upsample=4,
    # Except for this one, which I set to True
    use_scale_shift_norm=True,
)

model = model.cuda()

batch_size = 2
# This was an actual input
timesteps = tch.tensor([472.2500, 217.5000]).cuda()
assert timesteps.shape[0] == batch_size
# This is obviously not
x = tch.randn((batch_size, 3, 64, 64)).cuda()

model.eval()
model(x, timesteps)
