from pathlib import Path
import numpy as np
import torch
import gradio as gr
from PIL import Image
from model import TinyUNet

ROOT=Path(__file__).resolve().parents[1]
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MODEL=TinyUNet().to(DEVICE)
MODEL.load_state_dict(torch.load(ROOT/"models"/"best_restoration.pth",map_location=DEVICE))
MODEL.eval()

def restore(image):
    if image is None:
        return None,None,0
    img=Image.fromarray(image).convert("L")
    a=np.asarray(img,dtype=np.float32)/255

    with torch.no_grad():
        pred=MODEL(torch.tensor(a[None,None],dtype=torch.float32).to(DEVICE))[0,0].cpu().numpy()

    residual=np.abs(a-pred)
    score=(1-float(np.mean(residual)))*100
    return (pred*255).clip(0,255).astype(np.uint8), (residual*255).clip(0,255).astype(np.uint8), round(score,2)

demo=gr.Interface(
    fn=restore,
    inputs=gr.Image(type="numpy",image_mode="L",label="Degraded Inspection Image"),
    outputs=[
        gr.Image(type="numpy",image_mode="L",label="AI Restored"),
        gr.Image(type="numpy",image_mode="L",label="Residual Map"),
        gr.Number(label="Quality Indicator")
    ],
    title="Semiconductor Image Restoration AI",
    description="Educational PyTorch U-Net image restoration prototype."
)

if __name__=="__main__":
    demo.launch()
