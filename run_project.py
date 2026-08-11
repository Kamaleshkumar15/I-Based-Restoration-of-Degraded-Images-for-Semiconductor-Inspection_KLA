"""One-click IDLE launcher.
Open this file in IDLE and press F5.
"""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

def run(script):
    print("\n" + "="*60)
    print("RUNNING:", script.name)
    print("="*60)
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=False)

if __name__ == "__main__":
    print("AI Semiconductor Image Restoration")
    print("This launcher runs the complete pipeline.")
    run(SRC/"01_generate_dataset.py")
    run(SRC/"02_train_model.py")
    run(SRC/"03_evaluate_model.py")
    run(SRC/"04_restore_image.py")
    run(SRC/"05_create_gif.py")
    print("\nDONE. Open outputs/ and then run src/06_idle_app.py in IDLE.")
