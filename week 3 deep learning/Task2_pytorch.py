import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_tr, X_va, y_tr, y_va = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

scaler = StandardScaler().fit(X_tr)   # fit on train only
X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
print("Train:", X_tr.shape, "Val:", X_va.shape)
class CsvDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dl = DataLoader(CsvDataset(X_tr, y_tr), batch_size=32, shuffle=True)
val_dl = DataLoader(CsvDataset(X_va, y_va), batch_size=32)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x).squeeze(1)

model = MLP()
print("Params:", sum(p.numel() for p in model.parameters()))
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
def accuracy(m, dl):
    m.eval()                      # turn off dropout, use BatchNorm averages
    correct = 0
    with torch.no_grad():         # no gradients needed when just checking
        for xb, yb in dl:
            pred = (torch.sigmoid(m(xb)) > 0.5).float()
            correct += (pred == yb).sum().item()
    return correct / len(dl.dataset)
def accuracy(m, dl):
    m.eval()                      # turn off dropout, use BatchNorm averages
    correct = 0
    with torch.no_grad():         # no gradients needed when just checking
        for xb, yb in dl:
            pred = (torch.sigmoid(m(xb)) > 0.5).float()
            correct += (pred == yb).sum().item()
    return correct / len(dl.dataset)
best_acc, best_epoch, patience, wait = 0, 0, 8, 0
train_losses, val_accs = [], []

for epoch in range(1, 41):
    model.train()                 # dropout ON while training
    total = 0
    for xb, yb in train_dl:
        opt.zero_grad()           # 1. clear old gradients
        out = model(xb)           # 2. forward
        loss = loss_fn(out, yb)   # 3. loss
        loss.backward()           # 4. backward
        opt.step()                # 5. update weights
        total += loss.item() * len(xb)
    train_losses.append(total / len(train_dl.dataset))
    acc = accuracy(model, val_dl)
    val_accs.append(acc)
    print(f"epoch {epoch:2d} | loss {train_losses[-1]:.4f} | val acc {acc:.3f}")

    if acc > best_acc:            # new best: save it
        best_acc, best_epoch, wait = acc, epoch, 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:      # no improvement for 8 epochs
            print("Early stopping at epoch", epoch)
            break

print(f"Best val accuracy {best_acc:.2f} at epoch {best_epoch}")
new_model = MLP()
new_model.load_state_dict(torch.load("best_model.pt"))
new_model.eval()

reloaded_acc = accuracy(new_model, val_dl)
print(f"Reloaded model accuracy: {reloaded_acc:.2f}")
print("Identical to best:", abs(reloaded_acc - best_acc) < 1e-9)

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print("Logistic Regression accuracy:", round(lr_model.score(X_va, y_va), 2))
print("PyTorch MLP accuracy:", round(reloaded_acc, 2))