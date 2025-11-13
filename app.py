import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("CNN_model.keras", compile=False)
    return model

model = load_model()

# -------------------------------
# Correct CLASS_NAMES for 38 classes
# -------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

SUPPORTED_PLANTS = sorted(list(set([label.split("___")[0].replace("_(including_sour)", "") for label in CLASS_NAMES])))

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style='background-color:#2E7D32;padding:15px;border-radius:10px'>
    <h1 style='color:white;text-align:center;'>🌱 Plant Disease Detector</h1>
    <p style='color:white;text-align:center;font-size:16px;'>Upload or capture a leaf image to detect diseases.</p>
</div>
""", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🖼 Detect Disease", "🌾 Supported Plants"])

# -------------------------------
# Supported Plants Tab
# -------------------------------
with tab2:
    st.markdown("The model can detect diseases in the following plants:")
    cols = st.columns(3)
    for i, plant in enumerate(SUPPORTED_PLANTS):
        cols[i % 3].markdown(f"✅ **{plant}**")

# -------------------------------
# Leaf Detector Function
# -------------------------------
def looks_like_leaf(pil_image):
    img = np.array(pil_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / mask.size
    return green_ratio > 0.05

# -------------------------------
# Disease Detection Tab
# -------------------------------
with tab1:
    st.subheader("📸 Upload or Capture a Leaf Image")
    
    col1, col2 = st.columns(2)
    with col1:
        camera_image = st.camera_input("Take a photo")
    with col2:
        uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
    
    image_source = camera_image or uploaded_file

    if image_source:
        image = Image.open(image_source).convert("RGB")
        

        if not looks_like_leaf(image):
            st.warning("🚫 This image doesn’t appear to contain a leaf. Please upload a clear leaf photo.")
        else:
            img_resized = image.resize((244, 244))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            with st.spinner("🔍 Analyzing image..."):
                prediction = model.predict(img_array)

                if prediction.shape[1] != len(CLASS_NAMES):
                    st.error(f"⚠️ Model output ({prediction.shape[1]}) does not match CLASS_NAMES ({len(CLASS_NAMES)}).")
                else:
                    pred_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    disease_name = CLASS_NAMES[pred_index]

                    # ---------- DISPLAY RESULT ----------
                    st.markdown("### 🩺 Detection Result")
                    
                    if confidence < 0.6:
                        st.info("⚠️ Low confidence — please retake a clearer photo.")
                    elif "healthy" in disease_name.lower():
                        st.success(f"✅ The plant appears **HEALTHY** ({confidence:.2%} confidence)")
                    else:
                        st.error(f"🚨 Detected Disease: **{disease_name}**")
                        st.info(f"🧠 Model Confidence: {confidence:.2%}")

                    st.progress(float(confidence))

                    # ---------- TOP 3 PREDICTIONS ----------
                    top_3_indices = prediction[0].argsort()[-3:][::-1]
                    top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
                    top_3_probs = [float(prediction[0][i]) for i in top_3_indices]

                    st.markdown("### 🔎 Top 3 Predictions")
                    for cls, prob in zip(top_3_classes, top_3_probs):
                        st.info(f"**{cls}** → {prob*100:.2f}%")
    else:
        st.info("📷 Please capture or upload an image to begin detection.")
