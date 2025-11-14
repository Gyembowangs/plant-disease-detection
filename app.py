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

# Extract supported plants dynamically
SUPPORTED_PLANTS = sorted(list(set([label.split("___")[0].replace("_(including_sour)", "") for label in CLASS_NAMES])))

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

st.markdown("""
    <style>
        .title {
            color: white;
            text-align: center;
            background-color: #2E7D32;
            padding: 15px;
            border-radius: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            font-size: 16px;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌱 AI-Powered Plant Disease Detector</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture an image of a <b>plant leaf</b> to detect possible diseases using deep learning.</p>", unsafe_allow_html=True)

# -------------------------------
# Supported Plants Section
# -------------------------------
with st.expander("🌾 View Supported Plant Types"):
    st.markdown("The model can detect diseases in the following plants:")
    cols = st.columns(3)
    for i, plant in enumerate(SUPPORTED_PLANTS):
        cols[i % 3].markdown(f"✅ **{plant}**")

# -------------------------------
# Function: Improved Leaf Detector
# -------------------------------
def looks_like_leaf(pil_image):
    """Enhanced green filter: Return True if image likely contains a green leaf."""
    img = np.array(pil_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define strong green color range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Ratio of green pixels
    green_ratio = np.sum(mask > 0) / mask.size

    # Brightness check
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # A valid leaf image should have enough green area and moderate brightness
    return green_ratio > 0.10 and brightness > 60

# -------------------------------
# Image Input
# -------------------------------
st.subheader("📸 Upload or Capture a Leaf Image")

col1, col2 = st.columns(2)
with col1:
    camera_image = st.camera_input("Capture a leaf photo")
with col2:
    uploaded_file = st.file_uploader("Or upload a leaf image", type=["jpg", "jpeg", "png"])

image_source = camera_image or uploaded_file

# -------------------------------
# Process Image
# -------------------------------
if image_source:
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="📷 Input Image", use_container_width=True)

    if not looks_like_leaf(image):
        st.error("🚫 This image doesn’t appear to contain a leaf. Please upload a clear, green plant photo.")
    else:
        # Preprocessing
        img_resized = image.resize((244, 244))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        # Prediction
        with st.spinner("🔍 Analyzing image..."):
            prediction = model.predict(img_array)

            if prediction.shape[1] != len(CLASS_NAMES):
                st.error(f"⚠️ Model output ({prediction.shape[1]}) does not match CLASS_NAMES ({len(CLASS_NAMES)}).")
            else:
                pred_index = np.argmax(prediction)
                confidence = np.max(prediction)
                disease_name = CLASS_NAMES[pred_index]

                # Display Results
                st.subheader("🩺 Detection Result")
                if confidence < 0.6:
                    st.warning("⚠️ Low confidence — please retake a clearer photo.")
                elif "healthy" in disease_name.lower():
                    st.success(f"✅ The plant appears **HEALTHY** ({confidence:.2%} confidence)")
                else:
                    st.error(f"🚨 **Detected Disease:** {disease_name}")
                    st.info(f"🧠 Model Confidence: {confidence:.2%}")

                st.progress(float(confidence))

                # Top 3 Predictions
                st.subheader("🔎 Top 3 Predictions")
                top_3_indices = prediction[0].argsort()[-3:][::-1]
                top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
                top_3_probs = [float(prediction[0][i]) for i in top_3_indices]

                top3_df = pd.DataFrame({
                    "Class": top_3_classes,
                    "Confidence": [f"{p*100:.2f}%" for p in top_3_probs]
                })
                st.dataframe(top3_df, use_container_width=True)

else:
    st.info("📷 Please capture or upload an image to begin detection.")
