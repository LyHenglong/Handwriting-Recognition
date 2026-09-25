# Handwriting Recognition

A deep learning system that recognizes handwritten characters in real time. A custom ResNet-style CNN is trained on the EMNIST dataset (814K+ samples, 62 classes) and served through both a Jupyter demo and a standalone desktop application.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Getting Started](#getting-started)
- [How Prediction Works](#how-prediction-works)
- [Using the Model in Your Own Code](#using-the-model-in-your-own-code)
- [Roadmap](#roadmap)
- [License](#license)
- [Author](#author)

## Overview

This project trains a convolutional neural network to classify handwritten characters — digits (`0–9`), uppercase letters (`A–Z`), and lowercase letters (`a–z`) — and provides two ways to interact with it: a Jupyter notebook with a live drawing canvas, and a standalone Tkinter desktop app. It covers the full pipeline: data augmentation, model design, training with mixed precision, checkpointing, and inference-time image segmentation for multi-character input.

## Results

| Metric | Value |
|---|---|
| Dataset | EMNIST ByClass — 814,255 samples, 62 classes |
| Model | Custom ResNet CNN — 2.77M parameters |
| Test accuracy | **84.74%** |
| Training time | ~40 minutes (20 epochs on an RTX 5070 Ti) |

```
Epoch  1  →  Loss: 2.1500  |  Test Acc: 79.80%
Epoch  2  →  Loss: 1.9642  |  Test Acc: 81.94%
Epoch  6  →  Loss: 1.8744  |  Test Acc: 82.82%
Epoch  7  →  Loss: 1.8625  |  Test Acc: 83.77%
Epoch 11  →  Loss: 1.8290  |  Test Acc: 84.74%  ← best checkpoint
Epoch 20  →  Loss: 1.7908  |  Test Acc: 83.91%
```

Accuracy peaks around epoch 11 and drifts slightly afterward — expected behavior under cosine annealing. The best checkpoint is saved independently of the final one.

**Per-class error analysis** — the hardest characters are the ones that are visually ambiguous even to humans:

| Character | Accuracy | Likely confusion |
|---|---|---|
| `s` | 35.7% | `S` (case) |
| `l` | 35.8% | `1`, `I`, `i` |
| `o` | 43.3% | `0`, `O` |
| `O` | 47.0% | `0`, `o` |

## Demo

**Desktop app** (`handwriting_recognition_ui.py`) — a standalone Tkinter application with a drawing canvas, live prediction, confidence bar chart, prediction history, and model info panel.

> _Add a screenshot or GIF of the app here before sharing this repo._

**Notebook demo** (`draw_and_predict.ipynb`) — the same prediction pipeline inside Jupyter, for quick experimentation:

```
┌─────────────────────────┐
│                          │
│   Draw here with mouse   │  ← 280×280 canvas
│                          │
└─────────────────────────┘
  [ Clear ]  [ Predict ]
```

## Project Structure

```
Handwriting-Recognition/
├── handwriting_recognition_v2.ipynb   # Train the model
├── draw_and_predict.ipynb             # Jupyter drawing + prediction demo
├── handwriting_recognition_ui.py      # Standalone desktop app (Tkinter)
├── test_gpu.ipynb                     # Quick CUDA/GPU availability check
├── checkpoints/
│   ├── best_model.pth                 # Best saved weights
│   └── last_checkpoint.pth            # Resume training anytime
└── README.md
```

The EMNIST dataset is downloaded automatically on first run — no manual setup required.

## Model Architecture

```
Input (1×28×28 grayscale)
        │
   ┌────▼────┐
   │  Stem   │  Conv 3×3 → BatchNorm → ReLU          [32 ch, 28×28]
   └────┬────┘
        │
   ┌────▼────┐
   │ Layer 1 │  ResBlock(32→64, stride=2)             [64 ch, 14×14]
   │         │  ResBlock(64→64)
   └────┬────┘
        │
   ┌────▼────┐
   │ Layer 2 │  ResBlock(64→128, stride=2)            [128 ch, 7×7]
   │         │  ResBlock(128→128)
   └────┬────┘
        │
   ┌────▼────┐
   │ Layer 3 │  ResBlock(128→256, stride=2)           [256 ch, 4×4]
   │         │  ResBlock(256→256)
   └────┬────┘
        │
   Global Average Pool → Dropout(0.4) → Linear(256→62)
        │
   Output: 62 class logits
```

Each residual block uses a standard pre-activation-free design with a projection shortcut when the shape changes:

```
  Input ──────────────────────────────► (+) ──► ReLU
    │                                    ▲
    └─► Conv → BN → ReLU → Conv → BN ───┘
         (1×1 shortcut conv if shape changes)
```

## Training Setup

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 256 | Maximizes GPU utilization |
| Epochs | 20 | Sweet spot before overfitting |
| Optimizer | AdamW | Better weight decay decoupling than Adam |
| Learning rate | 0.001 → 1e-6 | Cosine annealing schedule |
| Weight decay | 1e-4 | L2 regularization |
| Loss | Cross-entropy + label smoothing (0.1) | Reduces overconfidence, handles class imbalance |
| Class weights | Inverse frequency | Upweights rare classes |
| Precision | Mixed precision (FP16 / AMP) | ~2× faster training, no accuracy loss |
| Dropout | 0.4 | Regularizes the classifier head |

**Data augmentation:** random rotation (±15°), random affine (10% translation, 5° shear), random erasing (p=0.2, 2–15% area), plus a fix for EMNIST's native mirrored/rotated orientation.

## Getting Started

### Prerequisites

```bash
pip install torch torchvision tqdm matplotlib numpy scipy pillow ipycanvas ipywidgets jupyterlab
```

A GPU is recommended but not required — the code falls back to CPU automatically.

### 1. Train the model

```bash
jupyter notebook handwriting_recognition_v2.ipynb
```

Run all cells. The best checkpoint is saved to `checkpoints/best_model.pth` automatically. If `last_checkpoint.pth` already exists, training resumes from it.

### 2. Run the desktop app

```bash
python handwriting_recognition_ui.py
```

### 3. Or try the notebook demo

```bash
jupyter notebook draw_and_predict.ipynb
```

Run all cells, draw a character on the canvas, and click **Predict**.

## How Prediction Works

1. The canvas image is captured.
2. Connected dark regions ("ink blobs") are located and treated as candidate characters.
3. Wide blobs (e.g. multiple characters written together) are split — first by looking for empty columns between letters, then by finding the thinnest point in the ink for touching/cursive strokes.
4. Each character crop is resized to 28×28, centered by center of mass (not just bounding box), and its strokes are thickened to match EMNIST's stroke width.
5. The model outputs a probability distribution over 62 classes; predictions below 30% confidence are shown as `?` instead of a guess.

## Using the Model in Your Own Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import string

classes = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )
    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))

class HandwritingResNet(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResidualBlock(32, 64, stride=2),   ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2),  ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).view(x.size(0), -1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = HandwritingResNet(62).to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.eval()

from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def predict(pil_image):
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5_p, top5_i = torch.topk(probs, 5)
    return [(classes[i], p.item()) for i, p in zip(top5_i, top5_p)]
```

**Common tweaks:**

| Goal | Change |
|---|---|
| Train longer | `EPOCHS = 30` in the training notebook |
| Larger batches | Increase `BATCH_SIZE` (VRAM-limited) |
| Different split | `split="byclass"` → `"balanced"` or `"letters"` |
| Digits only | `NUM_CLASSES = 10`, `split="digits"` |
| Export for deployment | `torch.onnx.export(model, dummy, "model.onnx")` |

## Roadmap

- [ ] Confidence calibration (model is sometimes overconfident on the `s`/`l` confusion pair)
- [ ] Word-level context via a language model to resolve ambiguous characters
- [ ] Mobile / touch support for the drawing canvas
- [ ] ONNX / TorchScript export for edge deployment
- [ ] Fine-tuning support on custom handwriting samples

## License

MIT — see [LICENSE](LICENSE).

## Author

**LyHenglong**
[GitHub](https://github.com/LyHenglong) · _add LinkedIn / portfolio / email here_

Built with PyTorch and the EMNIST dataset.
