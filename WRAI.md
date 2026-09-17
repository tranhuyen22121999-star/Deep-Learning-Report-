# AI / ML WORKING RULES & ARCHITECTURE GUIDELINES

> **System Directive:** Every ML project must strictly adhere to modular software architecture, explicit separation of concerns, complete reproducibility, and standard MLOps pipeline practices.

---

## 1. CORE AI/ML WORKING RULES

### Rule 1: Single Responsibility Principle (SRP)
- Each `.py` module must handle only one distinct component of the ML lifecycle.
- **Strict Separation:**
  - `config.py`: Central configurations and parameters only.
  - `dataset.py`: Raw data extraction, parsing, and directory structuring.
  - `data_loader.py`: Dataset wrappers, augmentations, and PyTorch DataLoaders.
  - `model.py`: Neural network architecture definitions and forward passes.
  - `train.py`: Training loops, optimization, loss tracking, and checkpointing.
  - `evaluate.py`: Performance metrics, confusion matrices, and report generation.
  - `utils.py`: Helper routines (reproducibility seeds, custom plots).
  - `run_pipeline.py`: Orchestration entry point running the full end-to-end pipeline.

### Rule 2: Absolute Reproducibility
- Set deterministic random seeds across all numerical/deep learning libraries:
  - Python `random`
  - `numpy`
  - `torch` (CPU & CUDA)
- Disable non-deterministic CuDNN benchmarking if required for exact reproducibility.

### Rule 3: Hyperparameter & Configuration Centralization
- **No Hardcoded Values:** Magic numbers, paths, batch sizes, learning rates, and image dimensions are strictly prohibited inside individual script files.
- Everything must be declared inside `config.py`.

### Rule 4: Standalone Execution & Modular Testing
- Every module must be independently executable for unit debugging.
- Include `if __name__ == "__main__":` blocks in all scripts to test isolate components (e.g., verifying tensor shapes in `model.py` or batch loading in `data_loader.py`).

---

## 2. REPOSITORY ARCHITECTURE

```text
fashion_mnist_project/
│
├── config.py             # Global configurations, paths, seeds, hyperparameters
├── dataset.py            # Data fetching, directory parsing, label mapping
├── data_loader.py        # Dataset class, transforms, and DataLoaders
├── model.py              # CNN / Architecture definitions
├── train.py              # Training loop, loss tracking, checkpoint saving
├── evaluate.py           # Evaluation metrics, confusion matrix, classification report
├── utils.py              # Helper routines (seeds, plotting, metrics visualization)
├── run_pipeline.py       # Main pipeline entry point
└── README.md             # Project documentation and execution guide
```

---

## 3. PIPELINE DATAFLOW

```text
               +-----------------------+
               |       config.py       |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |      dataset.py       |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |    data_loader.py     |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |       model.py        |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |       train.py        |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |      evaluate.py      |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |   run_pipeline.py     |
               +-----------------------+
```

---

## 4. COMPLETE MODULE SPECIFICATIONS & CODE IMPLEMENTATION

### 4.1 `config.py`
```python
import os
import torch

# Seed for Reproducibility
RANDOM_SEED = 42

# Dataset Configuration
DATASET_NAME = "andhikawb/fashion-mnist-png"

# Model & Image Parameters
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label Mapping
FASHION_MNIST_LABELS = {
    "0": "T-shirt/top",
    "1": "Trouser",
    "2": "Pullover",
    "3": "Dress",
    "4": "Coat",
    "5": "Sandal",
    "6": "Shirt",
    "7": "Sneaker",
    "8": "Bag",
    "9": "Ankle boot",
}

# Directories & Output Paths
OUTPUT_DIR = "./output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "fashion_mnist_cnn.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### 4.2 `dataset.py`
```python
import os
import glob
import pandas as pd
import kagglehub
import config

def download_and_parse_dataset():
    path = kagglehub.dataset_download(config.DATASET_NAME)
    png_files = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
    
    records = []
    for fp in png_files:
        rel = os.path.relpath(fp, path)
        parts = rel.split(os.sep)
        split = next((p for p in parts if p.lower() in ("train", "test", "val", "validation")), "unknown")
        label = parts[-2] if len(parts) >= 2 else "unknown"
        records.append({"filepath": fp, "split": split, "label": label})

    df = pd.DataFrame(records)
    df["label_name"] = df["label"].apply(lambda l: config.FASHION_MNIST_LABELS.get(str(l), str(l)))
    return df

if __name__ == "__main__":
    df = download_and_parse_dataset()
    print(f"Dataset Loaded Successfully! Total Samples: {len(df)}")
    print(df.head())
```

### 4.3 `data_loader.py`
```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config
from dataset import download_and_parse_dataset

class FashionMNISTPNGDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("L")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(df=None, batch_size=config.BATCH_SIZE):
    if df is None:
        df = download_and_parse_dataset()

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FashionMNISTPNGDataset(train_df, transform=transform)
    test_dataset = FashionMNISTPNGDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
```

### 4.4 `model.py`
```python
import torch
import torch.nn as nn

class FashionMNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNISTCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = FashionMNISTCNN()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print(f"Model Output Shape: {out.shape}")
```

### 4.5 `utils.py`
```python
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Fashion-MNIST Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.close()
```

### 4.6 `train.py`
```python
import torch
import torch.nn as nn
import torch.optim as optim
import config
from model import FashionMNISTCNN
from data_loader import get_data_loaders
from utils import set_seed

def train_model():
    set_seed(config.RANDOM_SEED)
    train_loader, test_loader = get_data_loaders()

    model = FashionMNISTCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Starting Training on Device: {config.DEVICE}")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved successfully to {config.MODEL_SAVE_PATH}")
    return model, test_loader

if __name__ == "__main__":
    train_model()
```

### 4.7 `evaluate.py`
```python
import torch
import numpy as np
import config
from model import FashionMNISTCNN
from data_loader import get_data_loaders
from utils import plot_confusion_matrix
from sklearn.metrics import classification_report

def evaluate_model(model=None, test_loader=None):
    if model is None or test_loader is None:
        _, test_loader = get_data_loaders()
        model = FashionMNISTCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = [config.FASHION_MNIST_LABELS[str(i)] for i in range(10)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm_path = f"{config.OUTPUT_DIR}/confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
```

### 4.8 `run_pipeline.py`
```python
import config
from train import train_model
from evaluate import evaluate_model

def main():
    print("=== Starting Fashion-MNIST AI Pipeline ===")
    model, test_loader = train_model()
    print("\n=== Evaluating Model ===")
    evaluate_model(model, test_loader)
    print("=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
```
