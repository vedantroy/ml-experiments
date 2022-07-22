import torch as th
from unet import ResBlock
from my_unet import MyResBlock

th.manual_seed(42)
# You will also need to set the env var
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# OR
# CUBLAS_WORKSPACE_CONFIG=:16:8
# From NVIDIA docs:
# > set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8"
# > (may limit overall performance) or ":4096:8"
# > (will increase library footprint in GPU memory by approximately 24MiB).
th.use_deterministic_algorithms(mode=True)


def get_shape(x):
    return x.weight.data.shape


def copy_weight(x, y):
    xs, ys = get_shape(x), get_shape(y)
    if xs != ys:
        raise ValueError(f"{xs} != {ys}")
    x.weight.data = y.weight.data.detach().clone().requires_grad_(True)


def test_res_block():
    pass
    # OpenAI res block params
    res_block_params = dict(
        channels=128,
        emb_channels=512,
        dropout=0.0,
        out_channels=128,
        use_conv=False,
        use_scale_shift_norm=True,
        dims=2,
        use_checkpoint=False,
    )

    x = th.randn((2, 128, 64, 64)).type(th.float32).cuda()
    t = th.randn((2, 512)).type(th.float32).cuda()

    block = ResBlock(**res_block_params).cuda()
    myblock = MyResBlock(
        res_block_params["channels"],
        res_block_params["out_channels"],
        res_block_params["emb_channels"],
    ).cuda()

    copy_weight(myblock.in_norm, block.in_layers[0])
    copy_weight(myblock.in_conv, block.in_layers[2])
    copy_weight(myblock.time_emb_linear, block.emb_layers[1])
    copy_weight(myblock.out_norm, block.out_layers[0])
    copy_weight(myblock.out_conv, block.out_layers[3])

    with th.no_grad():
        _x = x
        print(_x[0, 0, 0, 0])
        expected = block(x, t)
        actual = myblock(x, t)
        print(_x[0, 0, 0, 0])
        assert expected.shape == actual.shape
        print(expected[0, 0, 0, 0])
        print(actual[0, 0, 0, 0])
        # print(expected)


test_res_block()
