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
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def train(lr, epochs=5000, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros((1, 1))
    losses = []
    for _ in range(epochs):
        # forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)
        p = sigmoid(a1 @ W2 + b2)
        losses.append(cross_entropy(y, p))
        # backward pass
        dz2 = (p - y) / len(X)          # BCE + sigmoid shortcut
        dW2 = a1.T @ dz2;  db2 = dz2.sum(0, keepdims=True)
        dz1 = (dz2 @ W2.T) * (z1 > 0)   # ReLU derivative
        dW1 = X.T @ dz1;   db1 = dz1.sum(0, keepdims=True)
        # gradient descent step
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2
    return p, losses
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

rng = np.random.default_rng(1)
W1 = rng.normal(0, 1, (2, 4))
b1 = np.zeros((1, 4))
W2 = rng.normal(0, 1, (4, 1))
b2 = np.zeros((1, 1))
z1 = X @ W1 + b1
a1 = relu(z1)
p = sigmoid(a1 @ W2 + b2)
print("Guesses before training:", p.ravel().round(2))
print("Loss before training:", cross_entropy(y, p).round(3))
p, losses = train(0.5)
print("XOR predictions after training:", p.ravel().round(2))
print("Loss after training:", losses[-1].round(3))
import matplotlib.pyplot as plt

for lr in [0.01, 0.5, 5]:
    _, l = train(lr)
    plt.plot(l, label=f"lr={lr}")

plt.xlabel("epoch")
plt.ylabel("BCE loss")
plt.legend()
plt.savefig("loss_curves.png")
plt.show()