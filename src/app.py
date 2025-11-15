# app.py

import streamlit as st
from PIL import Image
import pandas as pd
from predict import predict_image

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(
    page_title="Orange Varieties Classification",
    page_icon="🍊",
    layout="wide"
)

st.title("🍊 Deep Learning for Orange Varieties Classification")
st.markdown("Upload one or more images of oranges to predict their varieties.")

# -----------------------------
# Upload Multiple Images
# -----------------------------
uploaded_files = st.file_uploader(
    "📷 Upload orange images (JPG/PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []

    # Display images side-by-side with predictions
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Predict
        predicted_class, confidence = predict_image(uploaded_file)
        results.append({
            "Image": uploaded_file.name,
            "Predicted Variety": predicted_class,
            "Confidence (%)": f"{confidence:.2f}"
        })

        # Side-by-side layout
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        with col2:
            st.markdown(f"**✅ Predicted Variety:** {predicted_class}")
            st.markdown(f"**📈 Confidence:** {confidence:.2f}%")

    # Display summary table
    st.markdown("### 📊 Predictions Summary")
    df_results = pd.DataFrame(results)
    st.table(df_results)

    # CSV download
    st.download_button(
        label="💾 Download Predictions as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.balloons()

else:
    st.info("👆 Upload one or more orange images to start classification.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
**Developed by Terna Henry Wua**  
B.Sc. Computer Science (2025) — J.S. Tarka University, Makurdi  
🔗 [GitHub](https://github.com/Clivez12) | [Portfolio](https://clivez12.github.io/My-Portfolio/)
""")
