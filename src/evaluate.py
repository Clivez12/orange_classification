import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "best_model.pth")
DATA_DIR = os.path.join(BASE_DIR, "dataset", "val")   # validation set
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
class_names = dataset.classes

# Load model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

# Run inference
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ---- Confusion Matrix ----
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print(f"✅ Confusion matrix saved to {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")

# ---- Classification Report ----
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(f"✅ Classification report saved to {report_path}")
print(report)

# ---- Sample Predictions ----
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis("off")

fig = plt.figure(figsize=(12, 8))
indices = np.random.choice(len(dataset), size=6, replace=False)
for i, idx in enumerate(indices):
    img, label = dataset[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        _, pred = torch.max(output, 1)
    ax = fig.add_subplot(2, 3, i+1)
    imshow(img, title=f"True: {class_names[label]}\nPred: {class_names[pred.item()]}")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"))
plt.close()

print(f"✅ Sample predictions saved to {os.path.join(RESULTS_DIR, 'sample_predictions.png')}")
