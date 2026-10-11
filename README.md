# Thermal DIP: Deep Image Prior for Thermal Image Restoration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likhith-naik01/DIP_MP/blob/main/ml/notebooks/colab_run.ipynb)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Restoring degraded thermal images (denoising high sensor noise and inpainting missing sensor pixels/dead regions) using **Deep Image Prior (DIP)** with a 5-level U-Net generator. 

**No training dataset is used.** The generator network is optimized at request time directly on the single degraded image, using the inductive bias of convolutional architectures to restore clean thermal radiation maps.

---

## What the System Does

1. **Input**: A degraded thermal infrared image (sensor noise, pixel dropout, scratches, or missing sensor block regions).
2. **Zero-Shot Optimization**: A fixed random noise tensor $z \sim \mathcal{N}(0, I)$ is passed through a 5-level U-Net generator $f_\theta$. The network weights $\theta$ are fitted using Adam to minimize the masked reconstruction loss:
   $$\mathcal{L} = \frac{1}{|M|} \| M \odot (f_\theta(z) - y) \|^2$$
   where $M$ is the binary valid pixel mask and $y$ is the degraded thermal frame.
3. **Restoration**: High-frequency thermal noise and corrupted holes are eliminated while preserving thermal gradients and edges.
4. **Benchmarking**: The restored output is benchmarked against classical filters (Mean 3x3, Mean 7x7, Median, Gaussian, OpenCV Inpaint) and Conv-DIP baseline, calculating PSNR and SSIM.

---

## Repository Structure

```
DIP_MP/
├── README.md                          # Project documentation
├── storage/
│   └── sample_images/                 # Pre-bundled thermal test scenes
│       ├── thermal_circuit_board.png
│       ├── thermal_building_heat_leak.png
│       └── thermal_machinery_bearing.png
│
├── ml/                                # Core Machine Learning package
│   ├── pyproject.toml                 # Package configuration
│   ├── setup.py                       # pip install -e ml/ setup
│   ├── dip_core/                      # Standalone DIP python package
│   │   ├── config.py                  # Hyperparameters dataclass
│   │   ├── metrics.py                 # PSNR and SSIM (skimage/torch)
│   │   ├── baselines.py               # Mean 3x3/7x7, Median, Gaussian filters
│   │   ├── models/
│   │   │   ├── unet.py                # 5-level U-Net with 1x1 skip projections
│   │   │   ├── conv_baseline.py       # Plain encoder-decoder without skips
│   │   │   └── blocks.py              # ConvBlock, DownBlock, UpBlock, Skip1x1
│   │   └── pipeline/
│   │       ├── preprocess.py          # Thermal loading, padding to multiple of 32, masks
│   │       ├── noise_input.py         # Noise z generation and regularization jitter
│   │       ├── losses.py              # Masked MSE loss
│   │       ├── trainer.py             # DIP optimization loop with callbacks
│   │       ├── early_stopping.py      # Loss plateau monitoring
│   │       └── postprocess.py         # Unpadding, FLIR colormaps, image saving
│   ├── scripts/
│   │   ├── run_single.py              # CLI: image in, restored image out
│   │   └── benchmark.py               # CLI: batch benchmark comparing DIP vs filters
│   ├── tests/
│   │   ├── test_unet_shapes.py        # U-Net tensor shapes across all 5 levels
│   │   ├── test_masked_loss.py        # Masked MSE loss mathematical validation
│   │   └── test_trainer_smoke.py      # 20-iteration optimization smoke test
│   └── notebooks/
│       ├── colab_run.ipynb            # Complete Google Colab GPU execution pipeline
│       └── experiments.ipynb          # Interactive analysis and visualization
│
├── backend/                           # Minimal FastAPI backend
│   ├── main.py                        # Single-file API: POST /restore, GET /health
│   └── requirements.txt               # Backend dependencies
│
└── frontend/                          # Zero-build HTML+JS frontend
    └── index.html                     # Single-page thermal studio with Chart.js
```

---

## 1. Running on Google Colab (Recommended for Free GPU)

Since Deep Image Prior trains a neural network per image, running on an **NVIDIA T4 GPU** in Google Colab restores images in **15 to 30 seconds** (compared to several minutes on CPU).

1. Click the **Open In Colab** badge above or open `ml/notebooks/colab_run.ipynb`.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run the 7 sequential cells:
   - Cell 1: Verifies CUDA GPU.
   - Cell 2: Clones the repository: `!git clone https://github.com/likhith-naik01/DIP_MP.git`.
   - Cell 3: Installs the package: `!pip install -e ml/`.
   - Cell 4: Mounts Google Drive for saving results (`/content/drive/MyDrive/DIP_outputs`).
   - Cell 5: Selects a sample thermal image or uploads your own.
   - Cell 6: Runs U-Net DIP and Conv baseline, plotting the restorations and loss curve.
   - Cell 7: Runs `benchmark.py` and exports `results.csv` and side-by-side comparison PNGs to Drive.

---

## 2. Running the ML Scripts Locally / CLI

### Install the ML Package
```bash
cd ml
pip install -e .
```

### Run on a Single Image
```bash
python scripts/run_single.py \
  --input ../storage/sample_images/thermal_circuit_board.png \
  --task denoise \
  --model unet \
  --iters 1500 \
  --output restored_circuit.png
```

To inpaint a damaged image with an optional binary mask:
```bash
python scripts/run_single.py \
  --input damaged_thermal.png \
  --mask defect_mask.png \
  --task inpaint \
  --model unet \
  --iters 1500 \
  --output inpainted_output.png
```

### Run Batch Benchmark
Compares U-Net DIP against Conv DIP, Mean 3x3, Mean 7x7, and Median filters:
```bash
python scripts/benchmark.py \
  --samples ../storage/sample_images \
  --output benchmark_output \
  --iters 1000 \
  --device cuda
```
Outputs `results.csv`, side-by-side comparison images, and `psnr_convergence_plot.png`.

### Run Unit Tests
```bash
python -c "import dip_core; from tests.test_unet_shapes import *; test_unet_output_shape(); test_unet_rectangular_shape(); print('Shapes OK'); from tests.test_masked_loss import *; test_masked_loss_full_mask(); print('Loss OK'); from tests.test_trainer_smoke import *; test_trainer_smoke_20_iterations(); print('Smoke test OK!')"
```

---

## 3. Starting the Backend

The backend is a lightweight FastAPI service that takes an image, optimizes the DIP network, and returns the restored base64 image and metrics.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- Health Check: `http://localhost:8000/health`
- Interactive API Docs: `http://localhost:8000/docs`

---

## 4. Opening the Frontend

The frontend is a self-contained, zero-build single HTML file (`frontend/index.html`).

Simply open `frontend/index.html` in your web browser:
```bash
# On Windows:
start frontend/index.html

# Or serve with Python:
python -m http.server 3000 --directory frontend
```

### Connecting Frontend to Colab
If you run the backend on Google Colab (using `cloudflared` or `ngrok`), simply paste the Colab public URL into the **Server URL** input at the top of the frontend interface!