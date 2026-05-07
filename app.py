import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

from utils.preprocess import preprocess_image
from utils.gradcam import make_gradcam_heatmap

# ---------------------------
# 🔧 LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras", compile=False)

model = load_model()

CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

SEVERITY_COLORS = {
    "No_DR": "#16A34A",
    "Mild": "#EAB308",
    "Moderate": "#F97316",
    "Severe": "#DC2626",
    "Proliferate_DR": "#7F1D1D"
}

INTERPRETATION = {
    "No_DR": "No signs of Diabetic Retinopathy detected.",
    "Mild": "Mild DR detected. Regular monitoring is recommended.",
    "Moderate": "Moderate DR detected. Medical attention is advised.",
    "Severe": "Severe DR detected. Immediate consultation required.",
    "Proliferate_DR": "Proliferative DR detected. Urgent treatment needed."
}

# ---------------------------
# 🎨 PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="DR Detection", layout="wide")

# ---------------------------
# 🧠 HEADER
# ---------------------------
st.title("🩺 Diabetic Retinopathy Detection")
st.write("Upload retinal images to detect severity and visualize Grad-CAM")

# ⚠️ Disclaimer
st.warning("⚠️ This tool is for educational purposes only and not a medical diagnosis.")

st.markdown("---")

# ---------------------------
# 📤 MULTIPLE UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload Fundus Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# ---------------------------
# 🚀 PROCESS EACH IMAGE
# ---------------------------
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):

        st.markdown("---")
        st.subheader(f"Processing: {uploaded_file.name}")

        image = Image.open(uploaded_file)
        img = np.array(image)

        # ✅ Input validation
        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            st.error("Invalid image. Please upload a proper retinal image.")
            continue

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # 🔧 Preprocess
        processed = preprocess_image(img)
        input_img = np.expand_dims(processed, axis=0)

        # 🔮 Prediction
        preds = model.predict(input_img)
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds))

        predicted_class = CLASS_NAMES[class_idx]
        color = SEVERITY_COLORS[predicted_class]

        with col2:
            # 🎯 Styled prediction
            st.markdown(f"""
            <div style="
                padding:15px;
                border-radius:10px;
                background-color:{color}20;
                border-left:6px solid {color};
            ">
            <b>Prediction:</b> {predicted_class}<br>
            <b>Confidence:</b> {confidence*100:.2f}%
            </div>
            """, unsafe_allow_html=True)

            # 🧠 Interpretation
            st.markdown("### 🧠 Interpretation")
            st.info(INTERPRETATION[predicted_class])

            # 📊 Probabilities
            st.markdown("### Class Probabilities")
            for i, prob in enumerate(preds[0]):
                if i == class_idx:
                    st.success(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")
                else:
                    st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

        # ---------------------------
        # 🔥 GRAD-CAM
        # ---------------------------
        heatmap = make_gradcam_heatmap(input_img, model)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.subheader("🔥 Grad-CAM")
        st.image(overlay, use_container_width=True)

        # ---------------------------
        # 📄 PDF REPORT
        # ---------------------------
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Diabetic Retinopathy Report", styles['Title']))
        content.append(Paragraph(f"Prediction: {predicted_class}", styles['Normal']))
        content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))
        content.append(Paragraph("<br/>Class Probabilities:", styles['Heading2']))

        for i, prob in enumerate(preds[0]):
            content.append(Paragraph(f"{CLASS_NAMES[i]}: {prob*100:.2f}%", styles['Normal']))

        content.append(Paragraph("<br/>Interpretation:", styles['Heading2']))
        content.append(Paragraph(INTERPRETATION[predicted_class], styles['Normal']))

        doc.build(content)

        # ✅ FIXED UNIQUE BUTTON
        st.download_button(
            label="📄 Download PDF Report",
            data=buffer.getvalue(),
            file_name=f"DR_Report_{idx}.pdf",
            mime="application/pdf",
            key=f"download_{idx}"
        )

# ---------------------------
# 🧠 HOW IT WORKS
# ---------------------------
with st.expander("ℹ️ How this works"):
    st.write("""
    - Model: EfficientNet-based deep learning model
    - Task: Multi-class classification of Diabetic Retinopathy
    - Input: Retinal fundus image
    - Output: Severity classification (5 classes)
    - Explainability: Grad-CAM highlights affected regions
    """)

# ---------------------------
# 📌 FOOTER
# ---------------------------
st.markdown("---")
st.markdown("<center>🚀 Built with Streamlit | Deep Learning Project</center>", unsafe_allow_html=True)