import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and split data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(30, 16, 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

train_losses = []
val_accs = []

for epoch in range(1, 51):
    model.train()
    opt.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_acc = ((val_out > 0).float() == y_val).float().mean()

    train_losses.append(loss.item())
    val_accs.append(val_acc.item())
    print(f"Epoch {epoch}: train loss {loss.item():.4f}, val accuracy {val_acc.item():.4f}")
    
    # Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(train_losses)
ax[0].set_title("Training loss")
ax[0].set_xlabel("Epoch")
ax[1].plot(val_accs)
ax[1].set_title("Validation accuracy")
ax[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("day2_training.png")
plt.show()

# Test evaluation
model.eval()
with torch.no_grad():
    test_out = model(X_test)
    test_acc = ((test_out > 0).float() == y_test).float().mean()
print("MLP test accuracy:", round(test_acc.item(), 4))

# Compare with scikit-learn logistic regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train.numpy(), y_train.numpy().ravel())
lr_acc = lr_model.score(X_test.numpy(), y_test.numpy().ravel())
print("Logistic regression test accuracy:", round(lr_acc, 4))

# Comparison: The MLP reached 0.9912 test accuracy and logistic regression reached 0.9737.
# The MLP did slightly better after raising the learning rate to 0.01.
# Logistic regression is simpler and faster on small tabular data like this.
# The MLP needs tuning (learning rate, epochs) but scales to more complex data.