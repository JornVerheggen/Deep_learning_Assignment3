import torch

b = 1
c = 2
h = 3
w = 4

X = torch.zeros(b, c, h, w)

Y1 = torch.max(torch.max(X, -1).values, -1)[0]
Y2 = X.mean(dim=-1).mean(dim=-1)

print(f"Input shape: {X.shape}, Output shape:{Y1.shape} for max")
print(f"Input shape: {X.shape}, Output shape:{Y2.shape} for mean")
