import torch

# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())
# x = x.reshape(3, 4)
# print(x)
#
# x = torch.zeros(2, 3, 4)
# print(x)
#
# x = torch.ones(2, 3, 4)
# print(x)
#
# x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# x = torch.exp(x)
# print(x)
#
# x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# y = torch.tensor([[2.0, 1.0, 4.0, 3.0], [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
# z1 = torch.cat((x, y), dim=0)
# z2 = torch.cat((x, y), dim=1)
# print(z1)
# print(z2)
#
# equals = x == y
# print(equals)
#
# x = x.sum()
# print(x)
#
# a = torch.arange(3).reshape(3, 1)
# b = torch.arange(2).reshape(1, 2)
# c = a + b
# print(a)
# print(b)
# print(c)
#
# x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# print(x)
# print(x[-1])
# print(x[1:3])
# print(x[:, -1])
# print(x[:, 1:3])
#
# A = x.numpy()
# B = torch.tensor(A)
# print(type(A))
# print(type(B))

# X = torch.arange(12).reshape(3, 4)
# print(len(X[0]))
# print(X.shape)

# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = A.clone()
# print(A)
# print(id(A) == id(B))
# C = A.reshape(2, 10)
# C[:] = 0
# print(C)
# print(id(A))
# print(id(C))

