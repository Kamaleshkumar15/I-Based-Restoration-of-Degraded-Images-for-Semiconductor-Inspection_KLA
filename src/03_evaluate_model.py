from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from model import TinyUNet

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"models"/"best_restoration.pth"
CLEAN = ROOT/"data"/"processed"/"clean"
BAD = ROOT/"data"/"processed"/"degraded"
OUT = ROOT/"outputs"
OUT.mkdir(exist_ok=True)

def psnr(a,b):
    mse = np.mean((a-b)**2)
    return 99.0 if mse == 0 else 10*np.log10(1/mse)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(MODEL,map_location=device))
    model.eval()

    names = sorted(BAD.glob("*.png"))[:50]
    p_before=[]; p_after=[]; s_after=[]

    for bad_path in names:
        clean_path = CLEAN/bad_path.name
        bad = np.asarray(Image.open(bad_path).convert("L"),dtype=np.float32)/255
        clean = np.asarray(Image.open(clean_path).convert("L"),dtype=np.float32)/255
        with torch.no_grad():
            pred = model(torch.tensor(bad[None,None],dtype=torch.float32).to(device))[0,0].cpu().numpy()

        p_before.append(psnr(bad,clean))
        p_after.append(psnr(pred,clean))
        s_after.append(ssim(clean,pred,data_range=1.0))

    report = (
        f"Samples evaluated: {len(names)}\n"
        f"Average PSNR before restoration: {np.mean(p_before):.2f} dB\n"
        f"Average PSNR after restoration:  {np.mean(p_after):.2f} dB\n"
        f"Average SSIM after restoration:  {np.mean(s_after):.4f}\n"
    )
    (OUT/"metrics.txt").write_text(report,encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
