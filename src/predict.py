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
SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")

MODEL_PATH = os.path.join(RESULTS_DIR, "model.pth")  # trained model
CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load class names
# -----------------------------
train_dir = os.path.join(DATA_DIR, "train")
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# -----------------------------
# Model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# Transform for prediction
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

    predicted_class = class_names[preds.item()]
    confidence = conf.item() * 100
    return predicted_class, confidence

def predict_folder(folder_path):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, file_name)
            prediction, confidence = predict_image(img_path)
            results.append({
                "Image": file_name,
                "Prediction": prediction,
                "Confidence (%)": f"{confidence:.2f}"
            })
    return results

# -----------------------------
# Run Tests
# -----------------------------
if __name__ == "__main__":
    # Single image prediction
    test_image = os.path.join(PROJECT_ROOT, "sample.jpg")  # put your image here
    if os.path.exists(test_image):
        prediction, confidence = predict_image(test_image)
        print(f"🍊 Single Image Prediction: {os.path.basename(test_image)} -> {prediction} ({confidence:.2f}%)")
    else:
        print(f"⚠️ Test image not found at {test_image}")

    # Folder prediction
    test_folder = os.path.join(PROJECT_ROOT, "test_images")  # put some images here
    if os.path.exists(test_folder):
        print("\n📂 Batch Predictions:")
        results = predict_folder(test_folder)
        for item in results:
            print(f"   {item['Image']} -> {item['Prediction']} ({item['Confidence (%)']}%)")
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n📑 Predictions saved to {CSV_PATH}")
    else:
        print(f"⚠️ Test folder not found at {test_folder}")
