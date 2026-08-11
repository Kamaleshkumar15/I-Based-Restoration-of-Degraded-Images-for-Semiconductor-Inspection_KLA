from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
from model import TinyUNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"/"processed"
MODEL_DIR = ROOT/"models"
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 16
LR = 0.001

class PairedDataset(Dataset):
    def __init__(self):
        self.clean = sorted((DATA/"clean").glob("*.png"))
        self.bad = sorted((DATA/"degraded").glob("*.png"))
        if not self.clean:
            raise RuntimeError("Dataset not found. Run 01_generate_dataset.py first.")

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        bad = np.asarray(Image.open(self.bad[i]).convert("L"), dtype=np.float32)/255
        clean = np.asarray(Image.open(self.clean[i]).convert("L"), dtype=np.float32)/255
        return torch.tensor(bad[None], dtype=torch.float32), torch.tensor(clean[None], dtype=torch.float32)

def main():
    ds = PairedDataset()
    n_val = max(1, int(len(ds)*0.15))
    n_train = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_train,n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss()

    best = float("inf")

    for epoch in range(1,EPOCHS+1):
        model.train()
        train_total = 0

        for x,y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()*x.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device),y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred,y).item()*x.size(0)

        train_loss = train_total/n_train
        val_loss /= n_val
        print(f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), MODEL_DIR/"best_restoration.pth")
            print("Saved best_restoration.pth")

if __name__ == "__main__":
    main()
