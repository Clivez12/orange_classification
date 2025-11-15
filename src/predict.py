# predict.py
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "orange_model.pth")
INPUT_DIR = os.path.join(PROJECT_ROOT, "input_images")
CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# Ensure folders exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------
# Load class names
# -----------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------
# Load Model
# -----------------------------
class OrangeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(OrangeNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

model = OrangeNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[preds.item()]
    confidence = conf.item() * 100
    return predicted_class, confidence

def predict_input_folder():
    results = []
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"⚠️ No images found in {INPUT_DIR}. Place your images there.")
        return

    for file_name in files:
        img_path = os.path.join(INPUT_DIR, file_name)
        pred_class, confidence = predict_image(img_path)
        results.append({
            "Image": file_name,
            "Prediction": pred_class,
            "Confidence (%)": f"{confidence:.2f}"
        })
        print(f"🍊 {file_name} -> {pred_class} ({confidence:.2f}%)")

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n📄 All predictions saved to: {CSV_PATH}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    predict_input_folder()
