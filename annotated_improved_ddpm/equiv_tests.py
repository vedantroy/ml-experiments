import torch as th
from openai.unet import AttentionBlock, QKVAttention, ResBlock
from my_unet import MyQKVAttention, MyResBlock, MyAttentionBlock

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


def same_storage(x, y):
    return x.storage().data_ptr() == y.storage().data_ptr()


def get_shape(x):
    return x.weight.data.shape


def clone_tensor(x):
    return x.detach().clone()


def copy_weight(x, y):
    xs, ys = get_shape(x), get_shape(y)
    if xs != ys:
        raise ValueError(f"{xs} != {ys}")
    x.weight.data = clone_tensor(y.weight.data)
    if hasattr(x, "bias"):
        x.bias.data = clone_tensor(y.bias.data)


def init_layer(x):
    th.nn.init.kaiming_uniform_(x.weight)
    if hasattr(x, "bias"):
        th.nn.init.constant_(x.bias, 0)


def test_res_block():
    # This test passes (check commit date)
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

    x = th.randn((2, 128, 64, 64)).type(th.float32).cuda()
    t = th.randn((2, 512)).type(th.float32).cuda()

    with th.no_grad():
        expected = block(x, t)
        actual = myblock(x, t)
        assert expected.shape == actual.shape
        assert not same_storage(x, expected)
        assert not same_storage(x, actual)

        assert (x == expected).all()
        assert (x == actual).all()

        # During initialization, the last convolution layer is all 0s
        # which makes the res block act like an identity layer
        # So, do random initialization for more rigorous testing
        init_layer(block.out_layers[3])
        copy_weight(myblock.out_conv, block.out_layers[3])

        dbg_a = {}
        dbg_b = {}
        expected = block(x, t, dbg_a)
        actual = myblock(x, t, dbg_b)
        assert (expected == actual).all()

def test_qkv():
    block = QKVAttention().cuda()
    myblock = MyQKVAttention().cuda()

    with th.no_grad():
        seq_len = 64 * 64
        model_dim = 128
        x = th.randn((2, 3 * model_dim, seq_len)).type(th.float32).cuda()
        y = block(x)
        y2 = myblock(x)
        assert (y == y2).all()

def test_attention():
    channels = 512
    block = AttentionBlock(channels, num_heads=2).cuda()
    myblock = MyAttentionBlock(channels, num_heads=2).cuda()

    x = th.randn((2, channels, 8, 8)).cuda()

    with th.no_grad():
        expected = block(x)
        actual = myblock(x)
        # This will be trivially true, since we initialize
        # the last convolution to all 0s
        assert (expected == actual).all()

        init_layer(myblock.proj_out)
        copy_weight(block.qkv, myblock.qkv)
        copy_weight(block.proj_out, myblock.proj_out)
        copy_weight(block.norm, myblock.norm)

        expected = block(x)
        actual = myblock(x)
        assert expected.shape == actual.shape
        assert (expected == actual).all()
    
test_res_block()
test_qkv()
test_attention()