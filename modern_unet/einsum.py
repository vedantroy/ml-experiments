import numpy as np


x = np.array([1, 2])
y = np.array([3, 3, 3])
z = np.einsum("i,j->ij", x, y)
assert z.shape == (2, 3)
assert (z == np.array([[3, 3, 3], [6, 6, 6]])).all()

z = np.einsum("i,j->ji", x, y)
assert z.shape == (3, 2)
assert (z == np.array([[3, 6], [3, 6], [3, 6]])).all()

z = np.einsum("i,j->i", x, y)
assert (z == np.array([9, 18])).all()

z = np.einsum("i,j->j", x, y)
assert (z == np.array([9, 9, 9])).all()


x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
x_ein = np.einsum("ijk->ij", x)

I, J, K = x.shape
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