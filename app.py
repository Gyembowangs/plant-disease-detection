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
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS for Mobile Responsiveness and Visual Appeal
# -------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header styling with glassmorphism */
    .header-container {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.95) 0%, rgba(76, 175, 80, 0.95) 50%, rgba(129, 199, 132, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), 
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .header-title {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.98);
        text-align: center;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Camera input styling for mobile */
    .stCameraInput > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .stCameraInput > div > div {
        width: 100% !important;
    }
    
    .stCameraInput video {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3),
                    0 0 0 3px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    
    .stCameraInput video:hover {
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4),
                    0 0 0 3px rgba(76, 175, 80, 0.5);
        transform: scale(1.01);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #4CAF50, #81C784) 1;
    }
    
    /* Result cards with glassmorphism */
    .result-card {
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.9) 0%, rgba(195, 207, 226, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15),
                    0 0 0 1px rgba(255, 255, 255, 0.2) inset;
        border-left: 5px solid #9e9e9e;
        animation: slideInLeft 0.5s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2),
                    0 0 0 1px rgba(255, 255, 255, 0.3) inset;
    }
    
    .healthy-card {
        background: linear-gradient(135deg, rgba(168, 230, 207, 0.95) 0%, rgba(220, 237, 193, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3),
                    0 0 0 1px rgba(255, 255, 255, 0.3) inset;
        border-left: 5px solid #4CAF50;
        animation: slideInRight 0.5s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .healthy-card::before {
        content: '✨';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 3rem;
        opacity: 0.3;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .healthy-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(76, 175, 80, 0.4),
                    0 0 0 1px rgba(255, 255, 255, 0.4) inset;
    }
    
    .disease-card {
        background: linear-gradient(135deg, rgba(255, 211, 182, 0.95) 0%, rgba(255, 170, 165, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(244, 67, 54, 0.3),
                    0 0 0 1px rgba(255, 255, 255, 0.3) inset;
        border-left: 5px solid #f44336;
        animation: slideInRight 0.5s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .disease-card::before {
        content: '⚠️';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 3rem;
        opacity: 0.3;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.1); opacity: 0.5; }
    }
    
    .disease-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(244, 67, 54, 0.4),
                    0 0 0 1px rgba(255, 255, 255, 0.4) inset;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Prediction list styling */
    .prediction-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(250, 250, 250, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1),
                    0 0 0 1px rgba(255, 255, 255, 0.2) inset;
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #4CAF50, #81C784) 1;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
        animation-fill-mode: both;
    }
    
    .prediction-item:nth-child(1) { animation-delay: 0.1s; }
    .prediction-item:nth-child(2) { animation-delay: 0.2s; }
    .prediction-item:nth-child(3) { animation-delay: 0.3s; }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-item:hover {
        transform: translateX(10px) scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15),
                    0 0 0 1px rgba(255, 255, 255, 0.3) inset;
    }
    
    /* Plant list styling */
    .plant-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 245, 245, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1),
                    0 0 0 1px rgba(255, 255, 255, 0.2) inset;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 4px solid #4CAF50;
    }
    
    .plant-item:hover {
        transform: translateX(10px) translateY(-3px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3),
                    0 0 0 1px rgba(255, 255, 255, 0.3) inset;
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.98) 0%, rgba(200, 230, 201, 0.98) 100%);
    }
    
    /* Image container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin: 1.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.01);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #81C784, #A5D6A7);
        border-radius: 10px;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .stCameraInput video {
            width: 100vw !important;
            max-width: 100vw !important;
            margin-left: -1rem;
            margin-right: -1rem;
            border-radius: 0 !important;
        }
        
        [data-testid="stCameraInput"] {
            width: 100% !important;
        }
        
        [data-testid="stCameraInput"] > div {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        .header-container {
            padding: 2rem 1.5rem;
            border-radius: 15px;
        }
        
        .result-card, .healthy-card, .disease-card {
            padding: 1.5rem;
            border-radius: 15px;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.9), rgba(129, 199, 132, 0.9)) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(100, 181, 246, 0.1) 100%);
        border-left: 4px solid #2196F3;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: rgba(255, 255, 255, 0.9);
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin: 2rem 0;
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4CAF50, #81C784);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header-container">
    <div class="header-title">🌱 Plant Disease Detector</div>
    <div class="header-subtitle">Upload or capture a leaf image to detect diseases with AI-powered analysis</div>
</div>
""", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🖼 Detect Disease", "🌾 Supported Plants"])

# -------------------------------
# Supported Plants Tab
# -------------------------------
with tab2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="font-size: 2.5rem; font-weight: 700; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-bottom: 0.5rem;">
            🌿 Supported Plants
        </h2>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.9);">
            Our AI model can detect diseases in the following plants
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, plant in enumerate(SUPPORTED_PLANTS):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="plant-item">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">🌿</span>
                    <strong style="font-size: 1.1rem; color: #2c3e50;">{plant}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="font-size: 2.5rem; font-weight: 700; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-bottom: 0.5rem;">
            📸 Upload or Capture Image
        </h2>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.9);">
            Choose your preferred method to analyze a leaf image
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;">
            <h3 style="color: white; font-size: 1.5rem; margin-bottom: 1rem;">📷 Camera Capture</h3>
            <p style="color: rgba(255,255,255,0.9);">Take a live photo with your device camera</p>
        </div>
        """, unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
    
    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;">
            <h3 style="color: white; font-size: 1.5rem; margin-bottom: 1rem;">📁 File Upload</h3>
            <p style="color: rgba(255,255,255,0.9);">Upload an image from your device</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    image_source = camera_image or uploaded_file

    if image_source:
        image = Image.open(image_source).convert("RGB")
        
        # Display the image
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="font-size: 2rem; font-weight: 700; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                📷 Your Image
            </h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded/Captured Image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not looks_like_leaf(image):
            st.markdown("""
            <div class="result-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 3rem;">🚫</span>
                    <h4 style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 0;">Image Validation Failed</h4>
                </div>
                <div style="background: rgba(255,255,255,0.5); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                    <p style="font-size: 1.2rem; color: #555; margin: 0;">
                        This image doesn't appear to contain a leaf. Please upload a clear leaf photo for accurate detection.
                    </p>
                </div>
                <p style="font-size: 1.1rem; color: #555; margin-top: 1rem;">
                    💡 <strong>Tip:</strong> Make sure the image shows a clear view of a plant leaf with good lighting.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            img_resized = image.resize((244, 244))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            with st.spinner("🔍 Analyzing image with AI model..."):
                prediction = model.predict(img_array, verbose=0)

                if prediction.shape[1] != len(CLASS_NAMES):
                    st.error(f"⚠️ Model output ({prediction.shape[1]}) does not match CLASS_NAMES ({len(CLASS_NAMES)}).")
                else:
                    pred_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    disease_name = CLASS_NAMES[pred_index]

                    # ---------- DISPLAY RESULT ----------
                    st.markdown("""
                    <div style="text-align: center; margin: 2.5rem 0 1.5rem 0;">
                        <h3 style="font-size: 2.5rem; font-weight: 700; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🩺 Detection Result
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence < 0.6:
                        st.markdown("""
                        <div class="result-card">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                <span style="font-size: 3rem;">⚠️</span>
                                <h4 style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 0;">Low Confidence Detection</h4>
                            </div>
                            <div style="background: rgba(255,255,255,0.5); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                                <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                                    <strong>Confidence Level:</strong> <span style="color: #f44336;">{:.2%}</span>
                                </p>
                            </div>
                            <p style="font-size: 1.1rem; color: #555; margin-top: 1rem;">
                                💡 <strong>Tip:</strong> Please retake a clearer photo with better lighting and focus for more accurate results.
                            </p>
                        </div>
                        """.format(confidence), unsafe_allow_html=True)
                    elif "healthy" in disease_name.lower():
                        st.markdown(f"""
                        <div class="healthy-card">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                <span style="font-size: 4rem;">✅</span>
                                <h3 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin: 0;">Plant is HEALTHY!</h3>
                            </div>
                            <div style="background: rgba(255,255,255,0.6); padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;">
                                <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                                    <strong>Confidence:</strong> <span class="confidence-badge">{confidence:.2%}</span>
                                </p>
                            </div>
                            <p style="font-size: 1.2rem; color: #2c3e50; margin-top: 1rem; font-weight: 500;">
                                🎉 Great news! Your plant appears to be in excellent health.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="disease-card">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                <span style="font-size: 4rem;">🚨</span>
                                <h3 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin: 0;">Disease Detected</h3>
                            </div>
                            <div style="background: rgba(255,255,255,0.6); padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;">
                                <p style="font-size: 1.6rem; font-weight: 700; color: #d32f2f; margin: 0.5rem 0;">
                                    {disease_name}
                                </p>
                                <p style="font-size: 1.2rem; margin-top: 1rem;">
                                    <strong>Model Confidence:</strong> <span style="color: #f44336; font-weight: 700;">{confidence:.2%}</span>
                                </p>
                            </div>
                            <div style="background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336; margin-top: 1rem;">
                                <p style="margin: 0; color: #d32f2f; font-weight: 600;">
                                    ⚠️ Please consult with a plant expert for treatment recommendations.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Confidence progress bar
                    st.markdown(f"""
                    <div style="margin: 2rem 0;">
                        <p style="font-size: 1.2rem; font-weight: 600; color: white; text-align: center; margin-bottom: 0.5rem;">
                            Confidence Level: <span style="color: #4CAF50;">{confidence:.2%}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(confidence))

                    # ---------- TOP 3 PREDICTIONS ----------
                    st.markdown("""
                    <div style="text-align: center; margin: 3rem 0 1.5rem 0;">
                        <h3 style="font-size: 2.5rem; font-weight: 700; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🔎 Top 3 Predictions
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    top_3_indices = prediction[0].argsort()[-3:][::-1]
                    top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
                    top_3_probs = [float(prediction[0][i]) for i in top_3_indices]

                    for idx, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs), 1):
                        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
                        st.markdown(f"""
                        <div class="prediction-item">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 2.5rem;">{emoji}</span>
                                <div style="flex: 1;">
                                    <p style="margin: 0; font-size: 1.3rem; font-weight: 700; color: #2c3e50;">
                                        {cls}
                                    </p>
                                </div>
                            </div>
                            <div style="background: linear-gradient(90deg, #4CAF50, #81C784); padding: 0.5rem 1rem; border-radius: 8px; margin-top: 0.5rem;">
                                <p style="margin: 0; color: white; font-weight: 600; font-size: 1.1rem;">
                                    Confidence: {prob*100:.2f}%
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 5rem; margin-bottom: 1rem;">📷</div>
            <h3 style="font-size: 2rem; font-weight: 700; color: white; margin-bottom: 1rem;">
                Ready to Detect Plant Diseases
            </h3>
            <p style="font-size: 1.3rem; color: rgba(255,255,255,0.9); margin-bottom: 2rem;">
                👆 Use the camera or file uploader above to get started
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.2); padding: 1rem 2rem; border-radius: 10px;">
                    <span style="font-size: 2rem;">📸</span>
                    <p style="margin: 0.5rem 0 0 0; font-weight: 600;">Camera</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem 2rem; border-radius: 10px;">
                    <span style="font-size: 2rem;">📁</span>
                    <p style="margin: 0.5rem 0 0 0; font-weight: 600;">Upload</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

