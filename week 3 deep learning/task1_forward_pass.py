import numpy as np
np.random.seed(1)

W1 = np.random.randn(3, 4); b1 = np.zeros(4)
W2 = np.random.randn(4, 1); b2 = np.zeros(1)

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

def forward(x):
    h = relu(x @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    return h, out

x = np.array([0.5, -1.2, 3.0])
h, out = forward(x)

print("W1:", W1.shape, "b1:", b1.shape)
print("W2:", W2.shape, "b2:", b2.shape)
print("Hidden:", h)
print("Output probability:", out)