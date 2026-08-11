from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT/"data"/"processed"/"clean"
DEGRADED = ROOT/"data"/"processed"/"degraded"

COUNT = 600
SIZE = 128

def make_clean(size, seed):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 8)
    d = ImageDraw.Draw(img)
    c = size//2
    r = int(size*0.44)

    d.ellipse((c-r,c-r,c+r,c+r), fill=82, outline=190, width=2)

    step = max(8, size//12)
    for x in range(c-r+5, c+r, step):
        d.line((x,c-r+5,x,c+r-5), fill=110, width=1)
    for y in range(c-r+5, c+r, step):
        d.line((c-r+5,y,c+r-5,y), fill=110, width=1)

    for rr in [int(r*.25), int(r*.48), int(r*.70), int(r*.88)]:
        d.ellipse((c-rr,c-rr,c+rr,c+rr), outline=140, width=1)

    for _ in range(180):
        x = int(rng.integers(c-r+4, c+r-4))
        y = int(rng.integers(c-r+4, c+r-4))
        if (x-c)**2 + (y-c)**2 < (r-4)**2:
            v = int(rng.integers(110, 210))
            d.rectangle((x,y,x+1,y+1), fill=v)

    a = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    a += 7*np.sin(xx/size*math.pi) + 5*np.cos(yy/size*math.pi)
    return Image.fromarray(np.clip(a,0,255).astype(np.uint8))

def degrade(img, seed):
    rng = np.random.default_rng(seed)
    x = img
    if rng.random() < .9:
        x = x.filter(ImageFilter.GaussianBlur(float(rng.uniform(.5,1.8))))
    if rng.random() < .85:
        x = ImageEnhance.Contrast(x).enhance(float(rng.uniform(.45,.85)))

    a = np.asarray(x, dtype=np.float32)
    h,w = a.shape
    yy,xx = np.mgrid[0:h,0:w]

    gx = rng.uniform(-.35,.35)
    gy = rng.uniform(-.35,.35)
    a *= 1 + gx*(xx-w/2)/w + gy*(yy-h/2)/h
    a += rng.normal(0, float(rng.uniform(6,28)), a.shape)

    if rng.random() < .7:
        temp = Image.fromarray(np.clip(a,0,255).astype(np.uint8))
        d = ImageDraw.Draw(temp)
        for _ in range(int(rng.integers(1,5))):
            y = int(rng.integers(5,h-5))
            d.line((0,y,w,y+rng.integers(-2,3)),
                   fill=int(rng.integers(0,256)), width=1)
        a = np.asarray(temp, dtype=np.float32)

    return Image.fromarray(np.clip(a,0,255).astype(np.uint8))

def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    DEGRADED.mkdir(parents=True, exist_ok=True)
    for i in range(COUNT):
        clean = make_clean(SIZE, i)
        bad = degrade(clean, 10000+i)
        clean.save(CLEAN/f"{i:04d}.png")
        bad.save(DEGRADED/f"{i:04d}.png")
    print(f"Created {COUNT} paired images.")
    print("Clean:", CLEAN)
    print("Degraded:", DEGRADED)

if __name__ == "__main__":
    main()
