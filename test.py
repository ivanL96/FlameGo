import numpy as np	
import time, sys

# [[4.0000]
#  [5.0000]
#  [6.0000]], shape=[3 1], dtype=float32, order=[0 1], strides=[1 1]) Tensor(
# [[1.0000]]

# a = np.array([4,5,6]).reshape((3,1))
# b = np.array([1]).reshape((1,1))
# c = a @ b
# print(a, b, c)
# sys.exit()

import torch

a = torch.tensor([[4, 5, 6]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([[1], [2], [3]], dtype=torch.float32, requires_grad=True)

c = torch.matmul(a, b)
print(c)
c.backward()

grad_a = a.grad
grad_b = b.grad

# print("Matrix a:\n", a)
# print("Matrix b:\n", b)
# print("Result matrix c (a @ b):\n", c)
print("Gradient of a (dc/da):\n", grad_a)
print("Gradient of b (dc/db):\n", grad_b)
