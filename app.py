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
# CLASS NAMES (38 classes)
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
    page_title="Plant Disease Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS - Glassmorphism + Light Gradients
# -------------------------------
st.markdown(
    """
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    :root{
      --glass-bg: rgba(255,255,255,0.6);
      --glass-strong: rgba(255,255,255,0.75);
      --accent: rgba(102,126,234,0.95);
      --accent-2: rgba(129,199,132,0.95);
      --muted: #5b6b73;
      --card-radius: 16px;
      --shadow: 0 8px 30px rgba(15,23,42,0.08);
    }

    *{ font-family: 'Inter', sans-serif; box-sizing: border-box; }

    /* App background (subtle) */
    .stApp {
      background: linear-gradient(180deg, #f5f7fb 0%, #eef6f7 50%, #f7fbff 100%);
      min-height: 100vh;
      padding-bottom: 40px;
    }

    /* Container */
    .main .block-container {
      padding-top: 22px;
      padding-left: 28px;
      padding-right: 28px;
      max-width: 1200px;
    }

    /* Header card */
    .header-card {
      background: linear-gradient(135deg, rgba(255,255,255,0.55), rgba(255,255,255,0.4));
      border-radius: 20px;
      padding: 28px;
      margin-bottom: 22px;
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.6);
      backdrop-filter: blur(8px) saturate(120%);
    }

    .app-title {
      font-size: 30px;
      font-weight: 800;
      color: #173b3f;
      margin: 0;
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .app-sub {
      margin: 6px 0 0 0;
      color: var(--muted);
      font-weight: 500;
    }

    /* Upload area */
    .input-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.62));
      border-radius: var(--card-radius);
      padding: 18px;
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.6);
      backdrop-filter: blur(6px);
    }

    .upload-opts {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }

    .upload-cta {
      padding: 10px 14px;
      border-radius: 12px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: white;
      font-weight: 700;
      box-shadow: 0 8px 18px rgba(102,126,234,0.16);
    }

    /* Image box */
    .image-container {
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(0,0,0,0.03);
      background: rgba(255,255,255,0.6);
      padding: 10px;
      box-shadow: var(--shadow);
    }

    /* Result cards */
    .result {
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.62));
      border-radius: 14px;
      padding: 18px;
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.6);
      backdrop-filter: blur(6px);
    }

    .healthy {
      border-left: 6px solid #4CAF50;
    }

    .disease {
      border-left: 6px solid #f44336;
    }

    .low-conf {
      border-left: 6px solid #ff9800;
    }

    .result h3 {
      margin: 0 0 8px 0;
      color: #153238;
    }

    .result p { margin: 6px 0; color: var(--muted); }

    /* Top predictions */
    .prediction-list {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }

    .pred-item{
      padding: 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(13,47,52,0.03);
      display:flex;
      align-items:center;
      gap: 12px;
      box-shadow: 0 6px 16px rgba(8,23,26,0.04);
    }

    .pred-badge {
      min-width: 64px;
      text-align:center;
      font-weight:700;
      padding: 8px 10px;
      border-radius: 10px;
      background: linear-gradient(90deg,#d7f0e6,#cfeee0);
      color: #0b3b2e;
    }

    /* Supported plants - clean UL / grid */
    .plants-wrap {
      background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.55));
      padding: 18px;
      border-radius: 14px;
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.6);
      backdrop-filter: blur(6px);
    }

    .plants-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      list-style: none;
      padding: 10px;
      margin: 0;
    }

    .plants-grid li{
      background: rgba(255,255,255,0.95);
      padding: 12px 14px;
      border-radius: 12px;
      display:flex;
      align-items:center;
      gap: 10px;
      font-weight: 600;
      color: #173b3f;
      border: 1px solid rgba(14,44,46,0.03);
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      cursor: default;
    }

    .plants-grid li:hover{
      transform: translateY(-6px);
      box-shadow: 0 12px 30px rgba(14,44,46,0.06);
    }

    /* Responsive */
@media (max-width: 1100px){
  .plants-grid { 
    grid-template-columns: repeat(3, 1fr); 
  }
}

@media (max-width: 800px){
  .plants-grid { 
    grid-template-columns: repeat(2, 1fr); 
  }
  .app-title { 
    font-size: 22px; 
  }
}

@media (max-width: 520px){

        /* Plants */
        .plants-grid { 
            grid-template-columns: 1fr; 
        }

        /* Upload Layout */
        .upload-opts { 
            flex-direction: column; 
            gap: 12px; 
            align-items: stretch; 
        }

        /* ===== MOBILE FIX FOR PREDICTION ITEMS ===== */

        .prediction-list {
            gap: 16px !important;
        }

        .pred-item {
            display: flex;
            flex-direction: column !important;
            align-items: flex-start !important;
            width: 100% !important;
            padding: 16px !important;
            text-align: left !important;
        }

        /* Make each internal div take full width */
        .pred-item > div {
            width: 100% !important;
        }

        /* Center the medal at top */
        .pred-item .medal {
            width: 100% !important;
            text-align: center !important;
            font-size: 2rem !important;
            margin-bottom: 6px !important;
        }

        /* Confidence badge full width */
        .pred-badge {
            width: 100% !important;
            text-align: center !important;
            margin-top: 10px !important;
            padding: 10px 0 !important;
            font-size: 1.1rem !important;
        }
        }

        /* Hide streamlit header/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER ----------
st.markdown(
    """
    <div class="header-card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
        <div>
          <div class="app-title">🌿 Plant Disease Detector</div>
          <div class="app-sub">AI-powered leaf analysis — upload or snap a photo to check plant health</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
          <div style="font-size:0.95rem; color:var(--muted);">Model: <strong>38 classes</strong></div>
          <div style="font-size:0.95rem; color:var(--muted);">Built for quick field checks</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["Detect", "Supported Plants"])

# -------------------------------
# Supported Plants Tab (clean UL grid)
# -------------------------------
with tab2:
    st.markdown(
        """
        <div style="display:flex; gap:16px; align-items:center; justify-content:space-between; flex-wrap:wrap; margin-bottom:12px;">
            <div>
                <h2 style="margin:0; color:#153238;">Supported Plants</h2>
                <div style="color:var(--muted); margin-top:6px;">Our model supports detection for the following plant types.</div>
            </div>
            <div style="color:var(--muted); font-weight:600;">Total: <strong>{count}</strong></div>
        </div>
        """.format(count=len(SUPPORTED_PLANTS)),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="plants-wrap">', unsafe_allow_html=True)
    # build the UL grid using HTML for maximum control
    plants_html = "<ul class='plants-grid'>"
    for p in SUPPORTED_PLANTS:
        plants_html += f"<li>🌱 &nbsp; {p}</li>"
    plants_html += "</ul>"
    st.markdown(plants_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Helper function: leaf detection heuristic
# -------------------------------
def looks_like_leaf(pil_image):
    try:
        img = np.array(pil_image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(mask > 0) / mask.size
        return green_ratio > 0.05
    except Exception:
        return True  # fallback to let model try if heuristic fails

# -------------------------------
# Detection Tab
# -------------------------------
with tab1:
    st.markdown(
        """
        <div style="display:flex; gap:16px; align-items:center; justify-content:space-between; margin-bottom:10px; flex-wrap:wrap;">
            <div>
                <h2 style="margin:0; color:#153238;">Upload or Capture Image</h2>
                <div style="color:var(--muted); margin-top:6px;">Use your camera for quick capture or upload an image (jpg, png).</div>
            </div>
            <div style="color:var(--muted); font-weight:600;">Tip: Ensure clear lighting and the leaf fills most of the frame.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; margin-bottom:10px;'>Camera</div>", unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; margin-bottom:10px;'>Upload</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    image_source = camera_image or uploaded_file

    if image_source:
        image = Image.open(image_source).convert("RGB")

        # show uploaded image in a styled container
        st.markdown('<div class="image-container" style="margin-top:18px;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # validate leaf-like
        if not looks_like_leaf(image):
            st.markdown('<div class="result" style="margin-top:18px;">', unsafe_allow_html=True)
            st.markdown('<h3>Image Check — Leaf Not Detected</h3>', unsafe_allow_html=True)
            st.markdown('<p>We could not detect enough green/leaf area in this photo. Please try a closer photo of a single leaf with better lighting.</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-weight:600; margin-top:8px;">Tip: place the leaf on a contrasting background (e.g., white paper) and avoid shadows.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # prepare image for model
            img_resized = image.resize((244, 244))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            with st.spinner("Analyzing image with AI model..."):
                prediction = model.predict(img_array, verbose=0)

            # safety check for mismatch
            if prediction.shape[1] != len(CLASS_NAMES):
                st.error(f"Model output shape ({prediction.shape[1]}) doesn't match CLASS_NAMES ({len(CLASS_NAMES)}).")
            else:
                pred_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                disease_name = CLASS_NAMES[pred_index]

                # top banner result
                if confidence < 0.6:
                    result_type = "low-conf"
                    header = "Low Confidence Detection"
                    sub = f"Confidence: {confidence:.2%} — image may be unclear."
                elif "healthy" in disease_name.lower():
                    result_type = "healthy"
                    header = "Plant appears HEALTHY"
                    sub = f"Confidence: {confidence:.2%}"
                else:
                    result_type = "disease"
                    header = "Disease Detected"
                    sub = f"{disease_name} — Confidence: {confidence:.2%}"

                st.markdown(f'<div class="result {result_type}" style="margin-top:18px;">', unsafe_allow_html=True)
                st.markdown(f'<h3>{header}</h3>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-weight:700; margin:6px 0 8px 0;">{sub}</p>', unsafe_allow_html=True)

                # helpful note
                if result_type == "healthy":
                    st.markdown('<p>🎉 Your plant looks healthy. Continue regular care — monitor for changes.</p>', unsafe_allow_html=True)
                elif result_type == "low-conf":
                    st.markdown('<p>⚠️ Try taking another photo with better focus and lighting. Make sure leaf occupies most of the frame.</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p>⚠️ This result suggests disease-like symptoms. Consult a local expert for diagnosis and treatment.</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Confidence progress
                st.markdown("<div style='margin-top:14px; margin-bottom:6px;'><strong>Confidence</strong></div>", unsafe_allow_html=True)
                st.progress(float(confidence))

                # Top 3 predictions
                st.markdown('<div style="margin-top:18px;">', unsafe_allow_html=True)
                st.markdown('<h3 style="margin-bottom:10px;">Top Predictions</h3>', unsafe_allow_html=True)

                top_3_indices = prediction[0].argsort()[-3:][::-1]
                top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
                top_3_probs = [float(prediction[0][i]) for i in top_3_indices]

                # Display prediction items
                st.markdown('<div class="prediction-list">', unsafe_allow_html=True)
                for idx, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs), start=1):
                    medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
                    pct = f"{prob*100:.2f}%"
                    pred_html = f"""
                        <div class="pred-item">
                          <div style="min-width:48px; text-align:center; font-weight:800; font-size:18px;">{medal}</div>
                          <div style="flex:1;">
                            <div style="font-weight:700; color:#153238;">{cls}</div>
                            <div style="font-size:13px; color:var(--muted); margin-top:4px;">Confidence: {pct}</div>
                          </div>
                          <div style="min-width:84px; text-align:right;">
                            <div style="background:linear-gradient(90deg,#e8f7ef,#d7f0e6); padding:6px 10px; border-radius:10px; font-weight:700;">{pct}</div>
                          </div>
                        </div>
                    """
                    st.markdown(pred_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty state card
        st.markdown(
            """
            <div style="margin-top:18px; padding:24px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.6)); box-shadow: 0 10px 30px rgba(15,23,42,0.06);">
                <div style="display:flex; gap:18px; align-items:center; justify-content:space-between; flex-wrap:wrap;">
                    <div>
                        <h3 style="margin:0; color:#153238;">Ready to detect plants</h3>
                        <div style="color:var(--muted); margin-top:6px;">Use the camera above for a quick check, or upload an image to get started.</div>
                    </div>
                    <div style="display:flex; gap:12px;">
                        <div style="padding:10px 14px; border-radius:10px; background:linear-gradient(90deg,var(--accent),var(--accent-2)); color:#fff; font-weight:700;">Get Started</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
