# ✍️ Handwriting Recognition — AI That Reads Your Writing

> **Draw a letter. The AI names it.** Trained on 700 000+ real handwritten characters, running live in your browser — no internet needed after setup.

---

## 🧠 What Is This?

This project trains a deep neural network to recognize **62 handwritten characters** — every digit (`0–9`), uppercase letter (`A–Z`), and lowercase letter (`a–z`) — and then lets you **draw on screen** and watch it predict in real time.

| | |
|---|---|
| **Dataset** | EMNIST ByClass — 814 255 samples |
| **Model** | Custom ResNet — 2.77 M parameters |
| **Best accuracy** | **84.74%** on the test set |
| **Hardware used** | NVIDIA RTX 5070 Ti (17.1 GB VRAM) |
| **Training time** | ~40 minutes (20 epochs) |

---

## 📁 Project Structure

```
ML project/
│
├── handwriting_recognition_v2.ipynb   ← Train the model
├── draw_and_predict.ipynb             ← Draw & predict live
│
├── checkpoints/
│   ├── best_model.pth                 ← Best saved weights
│   └── last_checkpoint.pth           ← Resume training anytime
│
└── data/                              ← EMNIST dataset (auto-downloaded)
```

---

## 🗂️ The Two Notebooks

### 📒 `handwriting_recognition_v2.ipynb` — Training

This is where the brain is built and trained.

**What it does, step by step:**

| Step | What happens |
|------|-------------|
| 1 | Loads the **EMNIST ByClass** dataset (auto-downloads on first run) |
| 2 | Applies data augmentation: random rotation, translation, shear, erasing |
| 3 | Fixes EMNIST's quirky image orientation (images arrive mirrored + rotated) |
| 4 | Builds a **ResNet-style CNN** with skip connections and batch normalization |
| 5 | Trains with **AdamW + Cosine LR scheduler + Mixed Precision (AMP)** |
| 6 | Saves checkpoints every epoch — resume training anytime if it crashes |
| 7 | Plots training loss and test accuracy curves |
| 8 | Shows per-class accuracy so you can see which characters are hardest |

**Hardest classes the model struggles with:**

| Char | Accuracy | Why it's hard |
|------|----------|---------------|
| `s` | 35.7% | Looks like `S` (case confusion) |
| `l` | 35.8% | Looks like `1`, `I`, `i` |
| `o` | 43.3% | Looks like `0`, `O` |
| `O` | 47.0% | Looks like `0`, `o` |

These are genuinely hard even for humans — context makes all the difference!

---

### 🖼️ `draw_and_predict.ipynb` — Live Demo

This is the fun part. Open this notebook, run all cells, and you get an **interactive drawing canvas** right inside Jupyter.

```
┌─────────────────────────┐
│                         │
│   Draw here with mouse  │  ← 280×280 canvas
│                         │
└─────────────────────────┘
  [ Clear ]  [ Predict ]
  Draw a character above, then click Predict.
```

**What happens when you click Predict:**

1. The canvas image is captured
2. **Character segmentation** finds each letter you drew (even if you wrote multiple)
3. Each character is preprocessed to match EMNIST's exact format
4. The model outputs a prediction with **top-5 confidence scores**
5. Results appear as a bar chart — green = confident, red = unsure (`?`)

**Smart segmentation features:**
- **Gap splitting** — finds empty space between well-separated letters
- **Valley splitting** — handles touching/cursive letters using ink density valleys
- **Stroke thickening** — compensates for thin canvas strokes vs. EMNIST's thick ones
- **Centre-of-mass centering** — places each character exactly as EMNIST expects

---

## 🏗️ Model Architecture

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

**Each ResidualBlock:**
```
  Input ──────────────────────────────► (+) ──► ReLU
    │                                    ▲
    └─► Conv → BN → ReLU → Conv → BN ───┘
         (if shape changes: 1×1 shortcut conv)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision tqdm matplotlib numpy scipy pillow ipycanvas ipywidgets jupyterlab
```

> **GPU recommended** but not required — the model will fall back to CPU automatically.

### Step 1 — Train the model

```bash
jupyter notebook handwriting_recognition_v2.ipynb
```

Run all cells. Training takes ~2 min/epoch on a modern GPU, longer on CPU.  
The best model is saved to `checkpoints/best_model.pth` automatically.

> **Already trained?** Skip to Step 2. The checkpoint resume logic will detect `last_checkpoint.pth` and pick up where it left off.

### Step 2 — Try it live

```bash
jupyter notebook draw_and_predict.ipynb
```

Run all cells → a canvas appears → draw a character → click **Predict**.

---

## ⚙️ Training Details

| Hyperparameter | Value | Why |
|---|---|---|
| Batch size | 256 | Maximizes GPU utilization |
| Epochs | 20 | Sweet spot before overfitting |
| Optimizer | AdamW | Better weight decay than Adam |
| Learning rate | 0.001 → 0.000001 | Cosine annealing schedule |
| Weight decay | 1e-4 | L2 regularization |
| Loss function | CrossEntropy + label smoothing 0.1 | Handles class imbalance |
| Class weights | Inverse frequency | Rare classes get more attention |
| Mixed precision | FP16 (AMP) | ~2× faster, same accuracy |
| Dropout | 0.4 | Prevents overfitting before classifier |

**Data augmentation applied during training:**

```
RandomRotation(±15°)
RandomAffine(translate=10%, shear=5°)
RandomErasing(p=0.2, area=2–15%)
EMNIST orientation fix (rot90 + flip)
```

---

## 📈 Training Results

```
Epoch  1  →  Loss: 2.1500  |  Test Acc: 79.80%  ⭐ new best
Epoch  2  →  Loss: 1.9642  |  Test Acc: 81.94%  ⭐ new best
Epoch  6  →  Loss: 1.8744  |  Test Acc: 82.82%  ⭐ new best
Epoch  7  →  Loss: 1.8625  |  Test Acc: 83.77%  ⭐ new best
Epoch 11  →  Loss: 1.8290  |  Test Acc: 84.74%  ⭐ new best  ← FINAL BEST
Epoch 20  →  Loss: 1.7908  |  Test Acc: 83.91%
```

The model hit peak accuracy at epoch 11 then slightly drifted — a normal pattern with cosine annealing. The best checkpoint is always saved separately.

---

## 💡 For Developers

### Checkpoint format

```python
{
    "epoch":           int,          # last completed epoch
    "model_state":     state_dict,   # model weights
    "optimizer_state": state_dict,   # AdamW state
    "scheduler_state": state_dict,   # cosine scheduler state
    "scaler_state":    state_dict,   # AMP GradScaler state
    "best_acc":        float,        # best test accuracy so far
    "train_losses":    list[float],  # per-epoch training loss
    "test_accuracies": list[float],  # per-epoch test accuracy
}
```

### Loading the model in your own code

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

# Load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = HandwritingResNet(62).to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.eval()

# Predict a 28×28 grayscale PIL image
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def predict(pil_image):
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5_p, top5_i = torch.topk(probs, 5)
    return [(classes[i], p.item()) for i, p in zip(top5_i, top5_p)]
```

### Extending the model

| Goal | What to change |
|------|---------------|
| More epochs | Set `EPOCHS = 30` in training notebook |
| Larger batch | Increase `BATCH_SIZE` (limited by VRAM) |
| Different split | Change `split="byclass"` to `"balanced"` or `"letters"` |
| Only digits | Set `NUM_CLASSES = 10`, use `split="digits"` |
| Export to ONNX | `torch.onnx.export(model, dummy, "model.onnx")` |

---

## 🔍 How the Live Prediction Works (Plain English)

1. **You draw** on a white 280×280 canvas with a thick black pen
2. The notebook **takes a photo** of the canvas
3. It **finds ink blobs** (connected dark regions) and treats each as a character
4. If a blob is too wide (e.g., you wrote "Hi"), it **splits it** by looking for empty columns between letters, or by finding the thinnest part of the ink
5. Each character crop gets **resized to 28×28**, centered by its center of mass (not just bounding box), and its strokes are thickened to match EMNIST
6. The model **outputs 62 probabilities** — the highest one wins
7. If the top confidence is below 30%, it shows `?` instead of guessing

---

## 🤝 Contributing

Pull requests welcome. Areas worth improving:

- [ ] Confidence calibration (model is sometimes overconfident on `s`/`l` confusion)
- [ ] Word-level context (use a language model to resolve ambiguous characters)
- [ ] Mobile / touch support for the canvas
- [ ] Export to TorchScript / ONNX for edge deployment
- [ ] Train on custom handwriting to personalize the model

---

## 📜 License

MIT — use it, modify it, build on it.

---

*Built with PyTorch · EMNIST · ipycanvas · trained on an RTX 5070 Ti*
