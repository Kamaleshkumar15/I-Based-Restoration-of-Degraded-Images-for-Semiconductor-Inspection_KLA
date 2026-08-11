# AI-Based Restoration of Degraded Images for Semiconductor Inspection

A U-Net based deep-learning project for restoring degraded semiconductor wafer inspection images.

> **Educational Prototype:** This project is inspired by industrial semiconductor inspection workflows. It uses synthetic wafer-style imagery and does **not** use proprietary industrial data, models, or algorithms.
https://github.com/python-pillow/Pillow/issues/5837
https://makeagif.com/gif/understanding-degradation-of-lithium-ion-batteries-the-university-of-oxford-N1MXDy
## 📌 Overview

Semiconductor inspection images can be affected by sensor noise, blur, low contrast, uneven illumination, and scan-line artifacts. These degradations can make circuit-pattern details harder to inspect.

This project builds an end-to-end image-restoration pipeline:

```text
Clean Wafer Image
       ↓
Artificial Degradation
       ↓
Noisy / Blurred Inspection Image
       ↓
U-Net Deep Learning Model
       ↓
Restored Image
       ↓
Residual / Difference Map
       ↓
PSNR / SSIM + Visual Inspection
```

The system includes both:

- 🖥️ A Tkinter desktop GUI that can be launched from Python IDLE
- 🌐 An optional Gradio browser interface

## ✨ Features

- Generate approximately **600 clean + degraded image pairs**
- Simulate multiple inspection-image degradations:
  - Sensor noise
  - Gaussian blur
  - Low contrast
  - Illumination variation
  - Scan-line artifacts
- Train a **U-Net** image-to-image restoration model
- Save the best model checkpoint
- Evaluate restoration using:
  - **PSNR**
  - **SSIM**
- Generate a pixel-wise **residual map**
- Restore user-supplied PNG, JPG, JPEG, or BMP images
- Launch a desktop application using **Tkinter**
- Launch an optional local web interface using **Gradio**
- Generate an animated restoration pipeline GIF

## 🧠 Model

The project uses a classic **U-Net encoder-decoder architecture** with skip connections.

The encoder extracts low- and mid-level features, the bottleneck captures a compact global representation, and the decoder reconstructs the restored image while reusing spatial details through skip connections.

Training uses pixel-wise reconstruction loss such as **MSE/L1** and the **Adam optimizer**.

### Default Training Configuration

| Parameter | Default |
|---|---:|
| Epochs | 8 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | MSE / L1 |

For a stronger college-project-quality result, the report recommends increasing training to approximately **30–50 epochs**.

## 📊 Evaluation

The project uses two standard image-quality metrics:

### PSNR

Peak Signal-to-Noise Ratio measures pixel-level fidelity. Higher values generally indicate that the restored image is closer to the clean reference.

### SSIM

Structural Similarity Index measures luminance, contrast, and structural similarity. Values closer to **1.0** indicate stronger structural similarity.

### Representative Results

| Metric | Value |
|---|---:|
| Samples evaluated | 50 |
| Average PSNR before restoration | 18.20 dB |
| Average PSNR after restoration | 26.80 dB |
| Average SSIM after restoration | 0.91 |

> These are representative/illustrative results from the project report. Actual values can vary depending on the generated dataset, random seed, and training configuration.

## 🔍 Residual Map

The residual map is calculated as:

```text
Residual Map = |Input Image - Restored Image|
```

Bright regions indicate areas where the model made larger corrections. This provides an additional visual signal for understanding where noise, blur, or artifacts were reduced.

## 🗂️ Project Structure

```text
semiconductor_inspection_restoration_IDLE/
│
├── README.md
├── requirements.txt
├── run_project.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── clean/
│       └── degraded/
│
├── models/
│   └── best_restoration.pth
│
├── outputs/
│   ├── metrics.txt
│   ├── restored.png
│   ├── residual.png
│   └── restoration_pipeline.gif
│
└── src/
    ├── model.py
    ├── 01_generate_dataset.py
    ├── 02_train_model.py
    ├── 03_evaluate_model.py
    ├── 04_restore_image.py
    ├── 05_create_gif.py
    ├── 06_idle_app.py
    └── 07_gradio_app.py
```

## 🛠️ Technologies Used

| Area | Technology |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Model | U-Net |
| Image Processing | OpenCV |
| Image Handling | Pillow |
| Numerical Computing | NumPy |
| Evaluation | scikit-image |
| Desktop GUI | Tkinter |
| Web UI | Gradio |
| Visualization | Matplotlib |
| Dataset | Synthetic wafer inspection images |

## 💻 Requirements

- Python **3.10 or newer**
- Python IDLE
- Windows, Linux, or macOS
- **4 GB+ RAM recommended**
- GPU is optional; it can speed up training

## 📦 Installation

Clone or extract the project and open a terminal inside the project folder.

Install the required packages:

```bash
pip install torch torchvision numpy pillow opencv-python scikit-image matplotlib tqdm
```

For the optional Gradio interface:

```bash
pip install gradio
```

## ▶️ Run the Complete Pipeline

The easiest option is to open:

```text
run_project.py
```

in Python IDLE and press:

```text
F5
```

This runs the complete workflow.

## 🔄 Run Step by Step

For more control, run the scripts in this order:

### 1. Generate Dataset

```bash
python src/01_generate_dataset.py
```

Generates approximately 600 clean/degraded image pairs.

### 2. Train U-Net

```bash
python src/02_train_model.py
```

Trains the restoration model and saves:

```text
models/best_restoration.pth
```

### 3. Evaluate Model

```bash
python src/03_evaluate_model.py
```

Calculates PSNR and SSIM and saves:

```text
outputs/metrics.txt
```

### 4. Restore an Image

```bash
python src/04_restore_image.py
```

Produces:

```text
outputs/restored.png
outputs/residual.png
```

### 5. Create Pipeline GIF

```bash
python src/05_create_gif.py
```

Produces:

```text
outputs/restoration_pipeline.gif
```

The animation shows:

```text
Clean Wafer
     ↓
Degraded Image
     ↓
AI Restored
     ↓
Residual Map
```

### 6. Launch Tkinter GUI

```bash
python src/06_idle_app.py
```

The GUI allows users to upload an image and view the input, restored output, and residual map.

### 7. Launch Gradio Web UI

```bash
python src/07_gradio_app.py
```

This launches a local browser-based interface for image restoration.

## 📥 Supported Input Images

The desktop and Gradio interfaces support:

- PNG
- JPG
- JPEG
- BMP

## 🧪 Testing & Validation

The project uses several validation approaches:

- Visual inspection of restored images
- Visual inspection of residual maps
- Held-out validation data
- Degradation sanity checks
- PSNR and SSIM evaluation
- End-to-end pipeline testing

## ✅ Advantages

- Restores multiple degradation types with one learned model
- Fully reproducible using synthetic data
- Simple Python/IDLE workflow
- Includes both quantitative and visual evaluation
- Provides an interpretable residual map
- Includes desktop and optional web interfaces

## ⚠️ Limitations

- Training and evaluation use **synthetic imagery**
- Performance on real wafer-inspection images has not been validated
- Synthetic degradation is only an approximation of real camera/hardware artifacts
- Only a relatively small U-Net architecture is used
- No alternative deep-learning architectures are benchmarked
- A residual map shows where corrections occurred, but does not by itself prove that every correction is correct
- Performance outside the degradation family represented in training is not guaranteed

## 🚀 Future Scope

Possible extensions include:

- Train using properly licensed/anonymized real wafer or PCB datasets
- Explore deeper U-Net variants
- Add attention gates
- Investigate transformer-based restoration models
- Use perceptual or GAN-based losses
- Extend residual mapping into automated anomaly/defect flagging
- Deploy the model as a lightweight inference service
- Benchmark against classical methods such as Wiener filtering and non-local means

## 🏭 Potential Applications

The restoration approach could be used as a preprocessing stage before automated optical inspection or as a quality-improvement tool for inspection engineers.

The same methodology can also be adapted to:

- Semiconductor inspection
- PCB inspection
- Textile inspection
- Microscopy image cleanup

## 🔒 Disclaimer

This project is an **independent educational prototype**. It is inspired by general industrial inspection workflows and does not use proprietary semiconductor inspection data, commercial inspection software, proprietary models, or company-internal algorithms.

## 📚 References

- Ronneberger, O., Fischer, P., Brox, T. — *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Wang, Z. et al. — *Image Quality Assessment: From Error Visibility to Structural Similarity*
- Goodfellow, I., Bengio, Y., Courville, A. — *Deep Learning*
- PyTorch Documentation
- scikit-image Documentation
- OpenCV Documentation

## 👨‍💻 Project Summary

**Project:** AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Approach:** U-Net Image-to-Image Restoration  
**Dataset:** Synthetic wafer inspection imagery  
**Evaluation:** PSNR + SSIM + Residual Map  
**Interfaces:** Tkinter + Optional Gradio  
**Purpose:** Educational demonstration of deep-learning-based inspection image restoration
