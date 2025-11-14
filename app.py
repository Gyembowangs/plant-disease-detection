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
# Custom CSS - Glass + Animations
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root{
  --accent: rgba(102,126,234,0.95);
  --accent-2: rgba(129,199,132,0.95);
  --muted: #5b6b73;
  --shadow: 0 8px 30px rgba(15,23,42,0.08);
  --card-radius: 16px;
}

/* General Styles */
*{ font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background: linear-gradient(180deg, #f5f7fb 0%, #eef6f7 50%, #f7fbff 100%); min-height:100vh; padding-bottom:40px; }
.main .block-container { padding-top:22px; padding-left:28px; padding-right:28px; max-width:1200px; }

/* Header */
.header-card { background: linear-gradient(135deg, rgba(255,255,255,0.55), rgba(255,255,255,0.4)); border-radius:20px; padding:28px; margin-bottom:22px; box-shadow: var(--shadow); border:1px solid rgba(255,255,255,0.6); backdrop-filter: blur(8px) saturate(120%); }
.app-title { font-size:30px; font-weight:800; color:#173b3f; margin:0; display:flex; gap:12px; align-items:center; }
.app-sub { margin:6px 0 0 0; color: var(--muted); font-weight:500; }

/* Input Card */
.input-card { background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.62)); border-radius: var(--card-radius); padding:18px; box-shadow: var(--shadow); border:1px solid rgba(255,255,255,0.6); backdrop-filter: blur(6px); }

/* Image Box */
.image-container { border-radius:14px; overflow:hidden; border:1px solid rgba(0,0,0,0.03); background: rgba(255,255,255,0.6); padding:10px; box-shadow: var(--shadow); }

/* Result */
.result { background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.62)); border-radius:14px; padding:18px; box-shadow: var(--shadow); border:1px solid rgba(255,255,255,0.6); backdrop-filter: blur(6px); margin-top:18px; }
.healthy { border-left:6px solid #4CAF50; }
.disease { border-left:6px solid #f44336; }
.low-conf { border-left:6px solid #ff9800; }
.result h3 { margin:0 0 8px 0; color:#153238; }
.result p { margin:6px 0; color: var(--muted); }

/* Confidence Bar */
.conf-bar { width:100%; height:16px; border-radius:12px; background:#e0e0e0; overflow:hidden; margin-top:6px; margin-bottom:12px; }
.conf-fill { height:100%; width:0%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius:12px; animation: fillBar 1s forwards; }
@keyframes fillBar { from {width:0%;} to {width:var(--width); } }

/* Prediction List */
.prediction-list { display:grid; grid-template-columns:1fr; gap:12px; margin-top:12px; }
.pred-item { padding:12px; border-radius:12px; background: rgba(255,255,255,0.95); border:1px solid rgba(13,47,52,0.03); display:flex; align-items:center; gap:12px; box-shadow: 0 6px 16px rgba(8,23,26,0.04); transition: transform 0.2s, box-shadow 0.2s; }
.pred-item:hover { transform: translateY(-4px); box-shadow:0 12px 30px rgba(14,44,46,0.08); }
.pred-badge { min-width:64px; text-align:center; font-weight:700; padding:8px 10px; border-radius:10px; background: linear-gradient(90deg,#d7f0e6,#cfeee0); color:#0b3b2e; }

/* Plants Grid */
.plants-wrap { background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.55)); padding:18px; border-radius:14px; box-shadow: var(--shadow); border:1px solid rgba(255,255,255,0.6); backdrop-filter: blur(6px); }
.plants-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; list-style:none; padding:10px; margin:0; }
.plants-grid li{ background:rgba(255,255,255,0.95); padding:12px 14px; border-radius:12px; display:flex; align-items:center; gap:10px; font-weight:600; color:#173b3f; border:1px solid rgba(14,44,46,0.03); transition: transform 0.18s ease, box-shadow 0.18s ease; cursor:default; }
.plants-grid li:hover { transform:translateY(-6px); box-shadow:0 12px 30px rgba(14,44,46,0.06); }

/* Responsive */
@media(max-width:800px){ .plants-grid{ grid-template-columns:repeat(2,1fr); } .app-title{ font-size:22px; } }
@media(max-width:520px){ .plants-grid{ grid-template-columns:1fr; } .pred-item{ flex-direction:column; align-items:flex-start; } }

/* Hide Streamlit Header/Footer */
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header-card">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
    <div>
      <div class="app-title">🌿 Plant Disease Detector</div>
      <div class="app-sub">AI-powered leaf analysis — upload or snap a photo</div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <div style="font-size:0.95rem; color:var(--muted);">Model: <strong>38 classes</strong></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["Detect", "Supported Plants"])

# -------------------------------
# Supported Plants Tab
# -------------------------------
with tab2:
    st.markdown(f"""
    <div style="display:flex; gap:16px; align-items:center; justify-content:space-between; flex-wrap:wrap; margin-bottom:12px;">
        <div>
            <h2 style="margin:0; color:#153238;">Supported Plants</h2>
            <div style="color:var(--muted); margin-top:6px;">Detection for the following plant types.</div>
        </div>
        <div style="color:var(--muted); font-weight:600;">Total: <strong>{len(SUPPORTED_PLANTS)}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="plants-wrap">', unsafe_allow_html=True)
    plants_html = "<ul class='plants-grid'>"
    for p in SUPPORTED_PLANTS:
        plants_html += f"<li>🌱 &nbsp; {p}</li>"
    plants_html += "</ul>"
    st.markdown(plants_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Leaf detection heuristic
# -------------------------------
def looks_like_leaf(pil_image):
    try:
        img = np.array(pil_image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([25,40,40]), np.array([90,255,255]))
        green_ratio = np.sum(mask>0)/mask.size
        return green_ratio>0.05
    except: return True

# -------------------------------
# Detection Tab
# -------------------------------
with tab1:
    st.markdown("""
    <div style="display:flex; gap:16px; align-items:center; justify-content:space-between; margin-bottom:10px; flex-wrap:wrap;">
        <h2 style="margin:0; color:#153238;">Upload or Capture Image</h2>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1,1])
    with left_col:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    image_source = camera_image or uploaded_file
    if image_source:
        image = Image.open(image_source).convert("RGB")
        st.markdown('<div class="image-container" style="margin-top:18px;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if not looks_like_leaf(image):
            st.warning("Leaf not detected clearly. Try another image with a clear leaf.")
        else:
            img_array = np.expand_dims(np.array(image.resize((244,244)))/255.0, axis=0)
            with st.spinner("Analyzing image..."):
                prediction = model.predict(img_array, verbose=0)

            pred_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            disease_name = CLASS_NAMES[pred_index]
            if confidence<0.6: result_type="low-conf"; header="Low Confidence"; sub=f"{confidence:.2%}"
            elif "healthy" in disease_name.lower(): result_type="healthy"; header="Healthy"; sub=f"{confidence:.2%}"
            else: result_type="disease"; header="Disease Detected"; sub=f"{disease_name} — {confidence:.2%}"

            st.markdown(f'<div class="result {result_type}"><h3>{header}</h3><p>{sub}</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="--width:{confidence*100}%"></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Top 3 predictions
            top3_idx = prediction[0].argsort()[-3:][::-1]
            st.markdown('<div class="prediction-list">', unsafe_allow_html=True)
            for idx, i in enumerate(top3_idx):
                cls = CLASS_NAMES[i]; prob = prediction[0][i]; medal = "🥇🥈🥉"[idx]
                st.markdown(f"""
                <div class="pred-item">
                    <div style="font-size:20px;">{"🥇" if idx==0 else "🥈" if idx==1 else "🥉"}</div>
                    <div style="flex:1;">
                        <div style="font-weight:700; color:#153238;">{cls}</div>
                        <div style="font-size:13px; color:var(--muted);">Confidence: {prob*100:.2f}%</div>
                    </div>
                    <div class="pred-badge">{prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Use the camera or upload an image to start detection.")
