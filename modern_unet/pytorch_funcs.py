import torch
import numpy as np

# Having the rainbow brackets extension installed helps you *understand* the code.
# Install it!
def chunk():
    x = torch.from_numpy(np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]))

    # 0th dim = row-wise
    a, b = torch.chunk(x, 2, dim=0)
    assert (a == torch.tensor([[[1, 2], [3, 4]]])).all()
    assert (b == torch.tensor([[[5, 6], [7, 8]]])).all()
    # 1st dim = col-wise
    a, b = torch.chunk(x, 2, dim=1)
    assert (a == torch.tensor([[[1, 2]], [[5, 6]]])).all()
    assert (b == torch.tensor([[[3, 4]], [[7, 8]]])).all()

    # This example is slightly more illustrative
    # We are splitting dim=1 into 2 chunks
    # So we iterate over each row (since dim=1 is rows) and then split it in half
    x = torch.from_numpy(np.array([
        [[1, 2], [3, 4], [9, 10], [13, 14]],
        [[5, 6], [7, 8], [11, 12], [15, 16]],
    ]))
    a, b = torch.chunk(x, 2, dim=1)
    assert (a == torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])).all()
    assert (b == torch.tensor([[[9, 10], [13, 14]], [[11, 12], [15, 16]]])).all()

chunk()