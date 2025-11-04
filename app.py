import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(
    page_title="Orange Varieties Classification",
    page_icon="🍊",
    layout="wide"
)

st.title("🍊 Deep Learning for Orange Varieties Classification")
st.markdown("Upload an image of an orange to predict its variety using a trained deep learning model.")

# -------------------------------
# Define Model (ResNet-based)
# -------------------------------
class OrangeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(OrangeNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
@st.cache_resource
def load_model():
    model = OrangeNet(num_classes=4)
    try:
        model.load_state_dict(torch.load("orange_model.pth", map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        st.error(f"⚠️ Model not found or failed to load: {e}")
    return model

model = load_model()

# -------------------------------
# Define Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels (update if needed)
CLASS_NAMES = ["Ibadan Sweet", "Tangerine", "Valencia", "Washington"]

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("📷 Upload an orange image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[preds.item()]

    # Display result
    st.success(f"✅ Predicted Variety: **{predicted_class}**")
    st.balloons()
else:
    st.info("👆 Upload an orange image to begin classification.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
---
**Developed by Terna Henry Wua**  
B.Sc. Computer Science (2025) — J.S. Tarka University, Makurdi  
🔗 [GitHub](https://github.com/Clivez12) | [Portfolio](https://clivez12.github.io/My-Portfolio/)
""")
