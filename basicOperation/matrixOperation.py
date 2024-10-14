import torch

# X = torch.arange(20).reshape(5, 4)
# print(X)
# Y = X.T
# print(Y)

# A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(A)
# print(A == A.T)

# X = torch.arange(24).reshape(2, 3, 4)
# print(X)
# print(X.T)

# X = torch.arange(20).reshape(5, 4)
# print(X)
# Y = X.clone()
# print(X*Y)
# a = 2
# print(X+a)
# print(X*a)

# X = torch.arange(20*2).reshape(2, 5, 4)
# print(X)
# print(X.sum())
# print(X.sum(axis=0))
# print(X.sum(axis=1))
# print(X.sum(axis=2))
# print(X.sum(axis=(0, 1)))

# X = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
# print(X)
# print(X.mean())
# print(X.numel())
# print(X.sum()/X.numel())

# X = torch.arange(20).reshape(5, 4)
# print(X)
# XSum = X.sum(axis=1, keepdim=True)
# print(XSum)
# print(X/XSum)
# print(X.cumsum(axis=0))

# a = torch.arange(4, dtype=torch.float32)
# # b = torch.ones(4)
# # print(a)
# # print(b)
# # print(torch.dot(a, b))
# X = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# Y = torch.ones(4, 3, dtype=torch.float32)
# # print(a)
# # print(X)
# # print(torch.mv(X, a))
# print(X)
# print(Y)
# print(torch.mm(X, Y))

# X = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(X)
# print(X.norm())
# print(torch.norm(X))
