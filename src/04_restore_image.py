from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import TinyUNet

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/"data"/"processed"/"degraded"/"0000.png"
OUT = ROOT/"outputs"
OUT.mkdir(exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(ROOT/"models"/"best_restoration.pth",map_location=device))
    model.eval()

    img = Image.open(INPUT).convert("L")
    a = np.asarray(img,dtype=np.float32)/255

    with torch.no_grad():
        pred = model(torch.tensor(a[None,None],dtype=torch.float32).to(device))[0,0].cpu().numpy()

    residual = np.abs(a-pred)

    Image.fromarray((pred*255).clip(0,255).astype(np.uint8)).save(OUT/"restored.png")
    Image.fromarray((residual*255).clip(0,255).astype(np.uint8)).save(OUT/"residual.png")

    print("Input:", INPUT)
    print("Saved:", OUT/"restored.png")
    print("Saved:", OUT/"residual.png")

if __name__ == "__main__":
    main()
