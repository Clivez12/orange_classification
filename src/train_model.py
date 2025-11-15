# train_model.py

import os
import json
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# ---------------------------------------------------
# Path Setup
# ---------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "orange_model.pth")
HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")
ACCURACY_PLOT = os.path.join(RESULTS_DIR, "accuracy_curve.png")
LOSS_PLOT = os.path.join(RESULTS_DIR, "loss_curve.png")

# ---------------------------------------------------
# Hyperparameters
# ---------------------------------------------------
NUM_CLASSES = 4
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# Data Transforms
# ---------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------------------------------
# Datasets and Loaders
# ---------------------------------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------
# Model Definition
# ---------------------------------------------------
class OrangeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = OrangeNet().to(DEVICE)

# ---------------------------------------------------
# Loss / Optimizer / Scheduler
# ---------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ---------------------------------------------------
# Training History
# ---------------------------------------------------
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
early_stop_counter = 0

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)

    # ----- Training -----
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = running_corrects.double() / len(train_dataset)

    # ----- Validation -----
    model.eval()
    val_loss_total = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss_total += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels)

    epoch_val_loss = val_loss_total / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)

    # Save history
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["train_acc"].append(epoch_train_acc.item())
    history["val_acc"].append(epoch_val_acc.item())

    print(f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.4f}")
    print(f"Val   Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.4f}")

    scheduler.step(epoch_val_loss)

    # Check best model
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, MODEL_PATH)
        print(f"🔥 Best model updated! | Val Acc = {best_acc:.4f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("⛔ Early stopping triggered!")
            break

# ---------------------------------------------------
# Save Training History
# ---------------------------------------------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=4)
print(f"\n📁 Training history saved to: {HISTORY_PATH}")

# ---------------------------------------------------
# Plot Curves
# ---------------------------------------------------
plt.figure()
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(ACCURACY_PLOT)
plt.close()

plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(LOSS_PLOT)
plt.close()

print(f"📊 Plots saved to: {RESULTS_DIR}")
print(f"🏁 Training complete! Best Validation Accuracy: {best_acc:.4f}")
