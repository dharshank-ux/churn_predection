import numpy as np
import matplotlib.pyplot as plt

# --- Softmax ---
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

p = softmax(np.array([2.0, 1.0, 0.1]))
print("softmax =", np.round(p, 3), "sum =", p.sum())

# --- Activation plots ---
x = np.linspace(-6, 6, 200)
plt.plot(x, np.maximum(0, x), label="ReLU")
plt.plot(x, 1 / (1 + np.exp(-x)), label="Sigmoid")
plt.plot(x, np.tanh(x), label="Tanh")
plt.legend(); plt.grid(True); plt.title("Activation functions")
plt.savefig("activations.png")
plt.show()

# --- Losses ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.2, 0.7])
print("MSE =", mse(y_true, y_pred))
print("Cross-entropy =", cross_entropy(y_true, y_pred))

# --- Confident wrong vs uncertain ---
print("Confident wrong (true=1, pred=0.01):", cross_entropy(np.array([1]), np.array([0.01])))
print("Uncertain (true=1, pred=0.5):", cross_entropy(np.array([1]), np.array([0.5])))
# --- Which activation and loss? ---
# 1. House price: linear output (no activation) + MSE, because it is a continuous number.
# 2. Spam yes/no: Sigmoid output + binary cross-entropy, because it is one probability.
# 3. 5-class image label: Softmax output + cross-entropy, because the 5 probabilities sum to 1.
# 4. Cross-entropy punishes confident wrong answers heavily, so it suits classification.