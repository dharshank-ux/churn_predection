import numpy as np

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def mse(y, p): return np.mean((y - p) ** 2)

def cross_entropy(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

print("softmax:", softmax(np.array([2.0, 1.0, 0.1])).round(3))
import torch

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x, t = 2.0, 10.0

loss = (w*x + b - t) ** 2
loss.backward()
print("Autograd: w.grad =", w.grad.item(), ", b.grad =", b.grad.item())