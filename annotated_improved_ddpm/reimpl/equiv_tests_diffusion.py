import torch as th

def test_linspace():
    # Get some intuition about linspace
    x = th.linspace(0, 4, steps=3)
    y = th.linspace(0, 3, steps=3)
    print(x, y)

test_linspace()