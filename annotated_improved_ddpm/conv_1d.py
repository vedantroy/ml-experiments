import torch as th


def test_conv_1d():
    IN, OUT, = (
        2,
        4,
    )
    KERNEL_SIZE = 1
    conv = th.nn.Conv1d(IN, OUT, KERNEL_SIZE)
    assert conv.weight.data.shape == (OUT, IN, KERNEL_SIZE)
    assert conv.bias.data.shape == (OUT,)

    ZERO_FILTER = [0]
    ONE_FILTER = [1]
    TWO_FILTER = [2]

    # Each filter is applied to 1 channel
    # And you sum row-wise (combining results from both channels)
    filters = th.Tensor(
        [
            [ZERO_FILTER, ZERO_FILTER],
            [ZERO_FILTER, ONE_FILTER],
            [ONE_FILTER, ZERO_FILTER],
            [TWO_FILTER, ONE_FILTER],
        ]
    )
    assert conv.weight.data.shape == filters.shape
    conv.weight.data = filters
    conv.bias.data = th.Tensor([0, 0, 0, 0])

    data = th.Tensor([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]])
    batch_size, c, seq = 1, IN, 6
    assert data.shape == (batch_size, c, seq)

    y = conv(data)
    assert y.shape == (batch_size, OUT, seq)

    eq = y == th.Tensor([
        [0, 0, 0, 0, 0, 0],
        [7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6],
        [2 * 1 + 1 * 7, 2 * 2 + 1 * 8, 2 * 3 + 1 * 9, 2 * 4 + 1 * 10, 2 * 5 + 1 * 11, 2 * 6 + 1 * 12],
    ])
    assert eq.all()

test_conv_1d()
