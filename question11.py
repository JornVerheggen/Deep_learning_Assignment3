import torch

b = 1
c = 2
h = 3
w = 4

X = torch.zeros(b, c, h, w)
Y1 = torch.nn.functional.max_pool2d(X, kernel_size=3)
Y2 = torch.nn.functional.avg_pool2d(X, kernel_size=3)


print(f"Input shape: {X.shape}, Output shape:{Y1.shape} for max")
print(f"Input shape: {X.shape}, Output shape:{Y2.shape} for mean")
