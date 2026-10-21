import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from Day3_task1_dataset import CsvDataset

def make_model():
    return nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 1))

def accuracy(m, dl):
    m.eval(); correct = 0
    with torch.no_grad():
        for x, y in dl:
            pred = torch.sigmoid(m(x)).squeeze(1) > 0.5
            correct += (pred == y).sum().item()
    return correct / len(dl.dataset)

if __name__ == '__main__':
    ds = CsvDataset('data.csv')
    train_ds, val_ds = random_split(ds, [455, 114], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    # train 30 epochs
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(30):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x).squeeze(1), y).backward()
            opt.step()
    acc_before = accuracy(model, val_dl)
    print("Validation accuracy BEFORE saving:", acc_before)

    # save
    torch.save(model.state_dict(), 'model.pt')

    # load into a NEW model
    new_model = make_model()
    new_model.load_state_dict(torch.load('model.pt'))
    new_model.eval()
    acc_after = accuracy(new_model, val_dl)
    print("Validation accuracy AFTER loading: ", acc_after)

    # predict one row
    def predict(features):
        new_model.eval()
        with torch.no_grad():
            p = torch.sigmoid(new_model(features.unsqueeze(0))).item()
        return int(p > 0.5), p

    label, prob = predict(ds[0][0])
    print(f"Predicted label: {label} | probability: {prob:.4f}")

# Why save only state_dict?
# It stores just the weights, so the file is small and portable. It does not depend on
# my class names or folder layout (saving the whole model uses pickle, which breaks if
# code is renamed or moved) and it is safer to load. I rebuild the architecture, then load weights.