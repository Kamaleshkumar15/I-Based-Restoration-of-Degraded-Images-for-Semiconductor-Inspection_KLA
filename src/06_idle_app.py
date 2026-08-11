"""IDLE-friendly restoration application.
Open in IDLE and press F5. No command-line arguments are required.
"""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
from PIL import Image, ImageTk
from model import TinyUNet

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT/"models"/"best_restoration.pth"

class App:
    def __init__(self, root):
        self.root=root
        root.title("Semiconductor Image Restoration AI")
        root.geometry("1050x650")

        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model=TinyUNet().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH,map_location=self.device))
        self.model.eval()

        tk.Label(root,text="AI-Based Restoration of Degraded Images",
                 font=("Arial",20,"bold")).pack(pady=15)

        tk.Button(root,text="Upload Degraded Image",command=self.restore,
                  font=("Arial",12,"bold")).pack(pady=8)

        self.info=tk.Label(root,text="Choose a grayscale inspection image.",
                           font=("Arial",11))
        self.info.pack()

        self.images=tk.Frame(root)
        self.images.pack(pady=20)

    def restore(self):
        path=filedialog.askopenfilename(
            title="Select degraded inspection image",
            filetypes=[("Image files","*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        img=Image.open(path).convert("L")
        a=np.asarray(img,dtype=np.float32)/255

        with torch.no_grad():
            pred=self.model(torch.tensor(a[None,None],dtype=torch.float32).to(self.device))[0,0].cpu().numpy()

        residual=np.abs(a-pred)

        restored=Image.fromarray((pred*255).clip(0,255).astype(np.uint8))
        residual_img=Image.fromarray((residual*255).clip(0,255).astype(np.uint8))

        for w in self.images.winfo_children():
            w.destroy()

        for title,im in [("Input",img),("AI Restored",restored),("Residual Map",residual_img)]:
            preview=im.resize((300,300))
            photo=ImageTk.PhotoImage(preview)
            panel=tk.Frame(self.images)
            panel.pack(side="left",padx=10)
            label=tk.Label(panel,image=photo)
            label.image=photo
            label.pack()
            tk.Label(panel,text=title,font=("Arial",12,"bold")).pack(pady=5)

        score=(1-float(np.mean(residual)))*100
        self.info.config(text=f"Restoration quality indicator: {score:.2f}% | Device: {self.device}")

if __name__=="__main__":
    root=tk.Tk()
    App(root)
    root.mainloop()
