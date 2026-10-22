import torch

A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = A @ B
print("(3x4) @ (4x2) ->", C.shape)
print("Device:", C.device)
import numpy as np

# Creating tensors: print shape, dtype, device
a = torch.tensor([[1, 2], [3, 4]])
z = torch.zeros(3, 4)
o = torch.ones(3, 4)
r = torch.arange(12)
n = torch.randn(3, 4)
for t in (a, z, o, r, n):
    print(t.shape, t.dtype, t.device)

# Slicing
print("first row:", n[0])
print("second column:", n[:, 1])
print("rows 1-2, cols 0-1:", n[1:3, :2])

# Reshape and transpose
print(r.view(3, 4))
print(r.reshape(4, 3))
print(n.T)

# NumPy interop
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
back = t.numpy()
print(t, back)

# GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
x = n.to(device)
print("Device:", x.device)