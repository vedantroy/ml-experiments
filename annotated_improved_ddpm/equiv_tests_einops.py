from einops import rearrange
import torch as th

# I was running into issues w/ einops, so these are some more
# tests for understanding


def test_chunk():
    test_chunk_impl(th.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).int())


def test_chunk_impl(a):
    b_t = a[..., None, None]
    b_e = rearrange(a, "b (c h w) -> b c h w", w=1, h=1)
    assert (b_t == b_e).all()

    c_t1, c_t2 = th.chunk(b_t, 2, dim=1)
    # INCORRECT
    # c_e1, c_e2 = rearrange(b_t, "b (split c) h w -> b split c h w", split=2)
    c_e1, c_e2 = rearrange(b_t, "b (split c) h w -> split b c h w", split=2)

    assert (c_t1 == c_e1).all()
    assert (c_t2 == c_e2).all()


test_chunk()
