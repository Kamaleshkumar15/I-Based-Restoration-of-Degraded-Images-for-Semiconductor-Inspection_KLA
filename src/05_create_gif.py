from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"
OUT.mkdir(exist_ok=True)

def label(img,title):
    img = img.convert("RGB").resize((384,384))
    canvas = Image.new("RGB",(384,425),"white")
    canvas.paste(img,(0,35))
    d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,384,35),fill="black")
    d.text((10,10),title,fill="white")
    return canvas

def main():
    clean = Image.open(ROOT/"data"/"processed"/"clean"/"0000.png").convert("L")
    bad = Image.open(ROOT/"data"/"processed"/"degraded"/"0000.png").convert("L")
    restored = Image.open(OUT/"restored.png").convert("L")
    residual = Image.open(OUT/"residual.png").convert("L")

    frames=[]
    for title,img in [
        ("CLEAN WAFER",clean),
        ("DEGRADED INSPECTION",bad),
        ("AI RESTORED",restored),
        ("RESIDUAL MAP",residual)
    ]:
        frames += [label(img,title)]*4

    path=OUT/"restoration_pipeline.gif"
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=450,loop=0)
    print("GIF created:",path)

if __name__ == "__main__":
    main()
