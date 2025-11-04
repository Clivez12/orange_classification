import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# --------------------------
# 1️⃣ PAGE CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="Orange Varieties Classifier",
    page_icon="🍊",
    layout="centered",
)

st.title("🍊 Orange Varieties Classification")
st.write("Upload an orange image to classify its variety using a deep learning model (PyTorch).")

# --------------------------
# 2️⃣ LOAD THE MODEL
# --------------------------
@st.cache_resource
def load_model():
    model_path = "saved_models/best_model.pth"
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes: Ibadan Sweet, Tangerine, Valencia, Washington
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# --------------------------
# 3️⃣ IMAGE PREPROCESSING
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["ibadan_sweet", "tangerine", "valencia", "washington"]

# --------------------------
# 4️⃣ IMAGE UPLOAD
# --------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --------------------------
    # 5️⃣ MAKE PREDICTION
    # --------------------------
    with st.spinner("Classifying... ⏳"):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        st.success(f"✅ Predicted Variety: **{classes[predicted.item()].capitalize()}**")
        st.write(f"Confidence: **{confidence.item() * 100:.2f}%**")

        # Show probability bar chart
        st.subheader("Prediction Confidence per Class")
        st.bar_chart({classes[i]: probabilities[i].item() for i in range(len(classes))})

# --------------------------
# 6️⃣ FOOTER
# --------------------------
st.markdown("---")
st.caption("Developed by **Terna Henry Wua** | B.Sc. Computer Science | Joseph Sarwuan Tarka University, Makurdi")
