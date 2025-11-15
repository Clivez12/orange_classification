# evaluate.py

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Paths
# -----------------------------
# Project root (one level above src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Results folder
RESULTS_DIR = PROJECT_ROOT / "results"

# Validation dataset folder
VAL_DIR = PROJECT_ROOT / "processed_dataset" / "val"

# Model path
MODEL_PATH = RESULTS_DIR / "orange_model.pth"

# Create results folder if not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not VAL_DIR.exists():
    raise FileNotFoundError(f"❌ Validation folder not found: {VAL_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Please train the model first.")

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------
# Class Names
# -----------------------------
CLASS_NAMES = sorted([d.name for d in (PROJECT_ROOT / "processed_dataset" / "train").iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------
# Data Loader
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -----------------------------
# Define Model
# -----------------------------
class OrangeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(OrangeNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load Model
# -----------------------------
model = OrangeNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# -----------------------------
# Run Evaluation
# -----------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = RESULTS_DIR / "confusion_matrix.png"
plt.tight_layout()
plt.savefig(cm_path)
plt.close()
print(f"✅ Confusion matrix saved to {cm_path}")

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
report_path = RESULTS_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"✅ Classification report saved to {report_path}")
print(report)

# -----------------------------
# Optional: Sample Predictions
# -----------------------------
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis("off")

fig = plt.figure(figsize=(12, 8))
indices = np.random.choice(len(val_dataset), size=min(6, len(val_dataset)), replace=False)
for i, idx in enumerate(indices):
    img, label = val_dataset[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(DEVICE))
        _, pred = torch.max(output, 1)
    ax = fig.add_subplot(2, 3, i + 1)
    imshow(img, title=f"True: {CLASS_NAMES[label]}\nPred: {CLASS_NAMES[pred.item()]}")
plt.tight_layout()
sample_pred_path = RESULTS_DIR / "sample_predictions.png"
plt.savefig(sample_pred_path)
plt.close()
print(f"✅ Sample predictions saved to {sample_pred_path}")
