import torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0)
Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
yva = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)

def make_net(reg=False):
    if reg:
        return nn.Sequential(
            nn.Linear(2, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))
    return nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

loss_fn = nn.BCEWithLogitsLoss()

def train(net, opt, epochs=100, patience=None):
    tr_losses, va_losses = [], []
    best, wait = float("inf"), 0
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xva), yva).item()
        tr_losses.append(loss.item()); va_losses.append(val)
        if patience:
            if val < best: best, wait = val, 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop at epoch {ep}"); break
    return tr_losses, va_losses

# Part A: compare 3 optimisers
configs = {
    "SGD": lambda p: torch.optim.SGD(p, lr=0.05),
    "SGD+momentum": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "Adam": lambda p: torch.optim.Adam(p, lr=0.01),
}
plt.figure()
for name, mk in configs.items():
    net = make_net()
    tr, va = train(net, mk(net.parameters()))
    plt.plot(tr, label=name)
plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend()
plt.savefig("optimiser_losses.png")

# Part B: with vs without Dropout + BatchNorm (Adam, early stopping)
plt.figure()
for name, reg in [("plain", False), ("Dropout+BatchNorm", True)]:
    net = make_net(reg)
    _, va = train(net, torch.optim.Adam(net.parameters(), lr=0.01), patience=10)
    plt.plot(va, label=name)
    print(name, "final val loss:", round(va[-1], 4))
plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.legend()
plt.savefig("regularisation_val_loss.png")
plt.show()