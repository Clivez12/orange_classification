# app.py

import streamlit as st
from PIL import Image
from predict import predict_image, predict_folder
import os
import pandas as pd

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(
    page_title="🍊 Orange Varieties Classification",
    page_icon="🍊",
    layout="wide"
)

st.title("🍊 Deep Learning for Orange Varieties Classification")
st.markdown("Upload an image or a folder of images to predict orange varieties.")

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_PARENT = os.path.dirname(PROJECT_ROOT)
FOLDER_DEFAULT = os.path.join(ROOT_PARENT, "new_images")

# -----------------------------
# Sidebar Options
# -----------------------------
option = st.sidebar.radio(
    "Select mode:",
    ("Single Image", "Folder Prediction")
)

# -----------------------------
# Single Image Mode
# -----------------------------
if option == "Single Image":
    uploaded_file = st.file_uploader("📷 Upload an orange image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict
        predicted_class, confidence = predict_image(uploaded_file)
        st.success(f"✅ Predicted Variety: **{predicted_class}** ({confidence:.2f}%)")
        st.balloons()
    else:
        st.info("👆 Upload an orange image to begin classification.")

# -----------------------------
# Folder Prediction Mode
# -----------------------------
elif option == "Folder Prediction":
    folder_path = st.text_input("📂 Folder path", value=FOLDER_DEFAULT)

    if st.button("Predict Folder"):
        if os.path.exists(folder_path) and os.listdir(folder_path):
            results = predict_folder(folder_path)

            # Display results
            st.write("📊 Prediction Results:")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            # Save CSV
            csv_path = os.path.join(ROOT_PARENT, "predictions_folder.csv")
            df_results.to_csv(csv_path, index=False)
            st.success(f"💾 All folder predictions saved to: {csv_path}")
        else:
            st.error("⚠️ Folder not found or empty.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
**Developed by Terna Henry Wua**  
B.Sc. Computer Science (2025) — J.S. Tarka University, Makurdi  
🔗 [GitHub](https://github.com/Clivez12) | [Portfolio](https://clivez12.github.io/My-Portfolio/)
""")
