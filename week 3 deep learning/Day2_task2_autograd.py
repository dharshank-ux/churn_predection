import torch

# Part 1: one backward pass
# Hand calculation: loss = (2w + b - 10)^2 = (-8)^2 = 64
# dloss/dw = 2 * (-8) * 2 = -32
# dloss/db = 2 * (-8) * 1 = -16
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
loss = (w * 2 + b - 10) ** 2
loss.backward()
print(f"w.grad = {w.grad.item()}, b.grad = {b.grad.item()}")

# Part 2: gradient descent for 30 steps
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01
for step in range(30):
    loss = (w * 2 + b - 10) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    print(step, loss.item())
print("final w:", w.item(), "final b:", b.item())

# Part 3: what happens if you forget to zero the gradients
w = torch.tensor(1.0, requires_grad=True)
for _ in range(2):
    ((w * 2 - 10) ** 2).backward()
    print("grad without zeroing:", w.grad.item())

# Part 4: explanation
# 1. requires_grad=True tells PyTorch to record operations on the tensor so it can compute gradients.
# 2. loss.backward() uses that record to fill the .grad of each such tensor.
# 3. detach() returns a tensor cut off from the graph, so no gradients flow through it.