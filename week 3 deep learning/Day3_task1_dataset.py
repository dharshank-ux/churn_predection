import torch, pandas as pd
from torch.utils.data import Dataset, DataLoader

class CsvDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.X = (X - X.mean(0)) / X.std(0)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

if __name__ == '__main__':
    ds = CsvDataset('data.csv')
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    xb, yb = next(iter(loader))
    print(f"Dataset size: {len(ds)} | batch features: {xb.shape} | batches: {len(loader)}")
    from torch.utils.data import random_split
    import time

    # split into train and validation
    train_ds, val_ds = random_split(ds, [455, 114], generator=torch.Generator().manual_seed(42))
    print("Train:", len(train_ds), "| Val:", len(val_ds))

    # count batches in 1 epoch
    count = 0
    for xb, yb in loader:
        count += 1
    print("Batches in 1 epoch:", count)

    # test num_workers 0 and 2
    for nw in [0, 2]:
        dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=nw)
        t = time.time()
        for xb, yb in dl:
            pass
        print(f"num_workers={nw}: {time.time()-t:.3f}s")
        
       