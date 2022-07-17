import numpy as np


def single_matrix():
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    I, J, K = x.shape
    x_ein = np.einsum("ijk->ij", x)

    x_loop = np.empty((I, J))
    for i in range(I):
        for j in range(J):
            total = 0
            for k in range(K):
                total += x[i, j, k]
            x_loop[i, j] = total
    assert (x_ein == x_loop).all()
    assert (x_ein == np.array([[3, 7], [11, 15]])).all()

    x_ein = np.einsum("ijk->k", x)
    x_loop = np.empty((K))
    for k in range(K):
        total = 0
        for i in range(I):
            for j in range(J):
                total += x[i, j, k]
        x_loop[k] = total
    assert (x_ein == x_loop).all()
    assert (x_ein == np.array([16, 20])).all()

    x_ein = np.einsum("ijk->ik", x)
    x_loop = np.empty((I, K))
    for i in range(I):
        for k in range(K):
            total = 0
            for j in range(J):
                total += x[i, j, k]
            x_loop[i, k] = total
    assert (x_ein == x_loop).all()
    assert (x_ein == np.array([[4, 6], [12, 14]])).all()

    x = np.array([[1, 2], [3, 4]])
    x_ein = np.einsum("ii->i", x)
    x_loop = np.empty((I))

    for i in range(I):
        x_loop[i] = x[i, i]
    assert (x_ein == x_loop).all()
    assert (x_ein == np.array([1, 4])).all()

    x = np.array([[1, 2], [3, 4]])
    x_ein = np.einsum("ii->", x)
    x_loop = np.empty((1))

    total = 0
    for i in range(I):
        total += x[i, i]
    x_loop[0] = total

    assert (x_ein == x_loop).all()
    assert (x_ein == np.array([5])).all()


def multiple_matrix():
    # multiple matrix einsum
    x = np.array([1, 2])
    (Xi,) = x.shape
    y = np.array([3, 3, 3])
    (Yj,) = y.shape

    z_ein = np.einsum("i,j->ij", x, y)
    z_loop = np.empty((Xi, Yj))
    for i in range(Xi):
        for j in range(Yj):
            z_loop[i, j] = x[i] * y[j]
    assert (z_ein == z_loop).all()

    z_ein = np.einsum("i,j->", x, y)
    z_loop = np.empty((1))
    total = 0
    for i in range(Xi):
        for j in range(Yj):
            total += x[i] * y[j]
    z_loop[0] = total
    assert (z_ein == z_loop).all()
    assert (z_ein == np.array([3 * 3 + 6 * 3])).all()

    # matrix multiplication
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[5, 6, 7, 8], [9, 10, 11, 12]])
    z_ein = np.einsum("ij,jk->ik", x, y)

    assert ((x @ y) == z_ein).all()
    I, J = x.shape
    _J, K = y.shape
    assert J == _J

    z_loop = np.empty((I, K))
    # free indices = i, k
    for i in range(I):
        # free indices always come first
        for k in range(K):
            total = 0
            # j = non-free index
            for j in range(J):
                total += x[i, j] * y[j, k]
            z_loop[i, k] = total
    assert (z_loop == z_ein).all()

    # The 1st part of attention (queries & keys)
    q = np.array(
        (
            [
                # single item in batch
                [
                    # 3 items in the sequence
                    [
                        # 2 heads
                        # d_q  = d_k = 3
                        [1, 2, 3],
                        [4, 5, 6],
                    ],
                    [[7, 8, 9], [10, 11, 12]],
                    [[13, 14, 15], [16, 17, 18]],
                ]
            ]
        )
    )
    B, I, H, D = q.shape
    k = np.array(
        (
            [
                # single item in batch
                [
                    # 3 items in the sequence
                    [
                        # 2 heads
                        # d_k = 3
                        [1, 2, 3],
                        [4, 5, 6],
                    ],
                    [[7, 8, 9], [10, 11, 12]],
                    [[13, 14, 15], [16, 17, 18]],
                ]
            ]
        )
    )
    assert q.shape == k.shape
    attn = np.einsum("bihd,bjhd->bijh", q, k)

    num_heads = H
    seq_len = I
    batch_size = B
    _, J, _, _ = k.shape
    assert I == J
    assert attn.shape == (batch_size, seq_len, seq_len, num_heads)
    attn_loop = np.empty((batch_size, seq_len, seq_len, num_heads))

    for b in range(B):
        for i in range(I):
            for j in range(J):
                for h in range(H):
                    total = 0
                    for d in range(D):
                        total += q[b, i, h, d] * k[b, j, h, d]
                    attn_loop[b, i, j, h] = total

    assert (attn_loop == attn).all()


    x = np.array([[1, 2], [3, 4], [5, 6]])
    A, B = x.shape
    y = np.array([[5, 6, 7, 8], [9, 10, 11, 12]])
    C, D = y.shape
    z_ein = np.einsum('ab,cd->ad', x, y)

    z_loop = np.empty((A, D))
    for a in range(A):
        for d in range(D):
            total = 0
            for b in range(B):
                for c in range(C):
                    total += x[a, b] * y[c, d]
            z_loop[a, d] = total

    assert (z_loop == z_ein).all()

    # Now, what does this print?
    # We know it's doing something for each row of x and each column of y
    # For each row @ the a-th idx of x and each column @ the d-th idx of y.transpose
    # For each value at the b-th idx of the row
    #   For each value at the c-th idx of the column
    #      muliply them together and add to the total

    assert (z_loop == np.array([
        [1 * 5 + 1 * 9 + 2 * 5 + 2 * 9, 1 * 6 + 1 * 10 + 2 * 6 + 2 * 10, 1 * 7 + 1 * 11 + 2 * 7 + 2 * 11, 1 * 8 + 1 * 12 + 2 * 8 + 2 * 12],
        [3 * 5 + 3 * 9 + 4 * 5 + 4 * 9, 3 * 6 + 3 * 10 + 4 * 6 + 4 * 10, 3 * 7 + 3 * 11 + 4 * 7 + 4 * 11, 3 * 8 + 3 * 12 + 4 * 8 + 4 * 12],
        [5 * 5 + 5 * 9 + 6 * 5 + 6 * 9, 5 * 6 + 5 * 10 + 6 * 6 + 6 * 10, 5 * 7 + 5 * 11 + 6 * 7 + 6 * 11, 5 * 8 + 5 * 12 + 6 * 8 + 6 * 12],
    ])).all()

single_matrix()
multiple_matrix()
