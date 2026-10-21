import matplotlib.pyplot as plt

x, t = 2.0, 10.0

# Hand gradient at start (w=1, b=0)
e = 1*x + 0 - t
print("dL/dw =", 2*e*x, " dL/db =", 2*e)   # -32, -16

def train(lr, steps=50):
    w, b = 1.0, 0.0
    losses = []
    for i in range(steps):
        e = w*x + b - t
        loss = e**2
        losses.append(loss)
        w -= lr * 2*e*x
        b -= lr * 2*e
        if i % 10 == 0:
            print(f"lr={lr} step {i}: loss={loss:.4f}")
    return losses

for lr in [0.001, 0.05, 0.5]:
    plt.plot(train(lr), label=f"lr={lr}")

plt.yscale("log")
plt.xlabel("step"); plt.ylabel("loss")
plt.legend()
plt.savefig("backprop_loss.png")
plt.show()
# A gradient is the slope of the loss with respect to each parameter.
# It points in the direction where the loss increases fastest.
# We subtract it so we move downhill, toward lower loss.
# The learning rate sets the step size: too big diverges, too small crawls.

# Epoch = one full pass through the training data.
# Batch size = number of samples used for one gradient update.