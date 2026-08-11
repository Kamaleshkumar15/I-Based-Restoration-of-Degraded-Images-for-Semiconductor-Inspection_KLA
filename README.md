# AI-Based Restoration of Degraded Images for Semiconductor Inspection

Educational ML/DL prototype inspired by semiconductor inspection workflows.

## What it does
1. Generates clean wafer-like inspection images.
2. Creates realistic synthetic degradation: noise, blur, low contrast, illumination variation and scan-line artifacts.
3. Trains a lightweight U-Net to restore degraded images.
4. Evaluates the model using L1 loss, PSNR and SSIM.
5. Generates a residual map for downstream inspection analysis.
6. Provides an IDLE-friendly GUI using Tkinter and an optional Gradio web UI.

## Important
This is an educational prototype. It does not use KLA proprietary data, algorithms or software and is not a production semiconductor inspection system.

## Run in Python IDLE
Open the project folder, then open each `.py` file in IDLE and press F5.

Recommended order:
1. `src/01_generate_dataset.py`
2. `src/02_train_model.py`
3. `src/03_evaluate_model.py`
4. `src/04_restore_image.py`
5. `src/05_create_gif.py`
6. `src/06_idle_app.py`

The included `models/best_restoration.pth` lets you try the restoration GUI immediately.

## Install packages
Use Command Prompt once:
`pip install torch torchvision numpy pillow opencv-python scikit-image matplotlib tqdm`

Optional Gradio:
`pip install gradio`

## Project flow
Clean image -> Synthetic degradation -> U-Net -> Restored image -> Residual map -> Quality metrics

## Suggested report sections
Problem statement, objectives, dataset generation, preprocessing, U-Net architecture, training, metrics, residual analysis, results, limitations and future work.
