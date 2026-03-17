import streamlit as st
import numpy as np
from PIL import Image
import base64

# ---- Background Image Function ----
def set_bg():
    path = r"C:\Users\satka\OneDrive\Pictures\Screenshots\bg1.jpeg"

    with open(path, "rb") as f:
        data = f.read()

    encoded = base64.b64encode(data).decode()

    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)

# ---- Robust model loader ----
def _get_loader():
    try:
        from tf_keras.models import load_model
        return load_model, "tf_keras"
    except Exception:
        from tensorflow.keras.models import load_model
        return load_model, "tf.keras"

load_model, backend = _get_loader()

st.set_page_config(
    page_title="IMAGE DEEPFAKE DETECTION",
    page_icon="🔍",
    layout="wide"
)

# ---- Set Background ----
set_bg()

# -------- Custom Styling --------
st.markdown("""
<style>
.big-title {
    font-size:55px !important;
    font-weight:900;
    margin-bottom:15px;
    color:white;
}
.subtitle {
    font-size:22px;
    margin-bottom:35px;
    color:white;
}
.result-box {
    padding:35px;
    border-radius:20px;
    text-align:center;
    font-size:32px;
    font-weight:900;
    margin-bottom:15px;
}
.real {
    background-color:#d4edda;
    color:#155724;
}
.fake {
    background-color:#f8d7da;
    color:#721c24;
}
.conf-text {
    font-size:22px;
    font-weight:700;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -------- Title --------
st.markdown("<div class='big-title'>🔎 IMAGE DEEPFAKE DETECTION</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload two images and compare predictions.</div>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_xception_model():
    return load_model("xception_model.h5", compile=False)

IMG_H, IMG_W = 128, 128
RESCALE = 1.0/255.0

try:
    model = load_xception_model()
except Exception as e:
    st.error("Error loading model.")
    st.exception(e)
    st.stop()

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_W, IMG_H))
    arr = np.asarray(img, dtype=np.float32) * RESCALE
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------- Two Uploaders --------
col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("📤 Upload Left Image", type=["jpg","jpeg","png","webp"], key="left")

with col2:
    file2 = st.file_uploader("📤 Upload Right Image", type=["jpg","jpeg","png","webp"], key="right")

# -------- Show Images --------
img_col1, img_col2 = st.columns(2)

pil1, pil2 = None, None

if file1:
    pil1 = Image.open(file1).convert("RGB")
    img_col1.image(pil1, caption="Left Image", width=350)

if file2:
    pil2 = Image.open(file2).convert("RGB")
    img_col2.image(pil2, caption="Right Image", width=350)

# -------- Prediction --------
if pil1 or pil2:
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    # Left Prediction
    if pil1:
        x1 = preprocess_pil(pil1)
        y1 = model.predict(x1, verbose=0)
        prob1 = float(y1[0][0])
        label1 = "Real" if prob1 >= 0.5 else "Fake"
        confidence1 = prob1 if label1 == "Real" else 1 - prob1

        with result_col1:
            if label1 == "Real":
                st.markdown("<div class='result-box real'>LEFT: ✅ REAL</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box fake'>LEFT: ❌ FAKE</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='conf-text'>Confidence: {confidence1*100:.2f}%</div>", unsafe_allow_html=True)
            st.progress(float(confidence1))

    # Right Prediction
    if pil2:
        x2 = preprocess_pil(pil2)
        y2 = model.predict(x2, verbose=0)
        prob2 = float(y2[0][0])
        label2 = "Real" if prob2 >= 0.5 else "Fake"
        confidence2 = prob2 if label2 == "Real" else 1 - prob2

        with result_col2:
            if label2 == "Real":
                st.markdown("<div class='result-box real'>RIGHT: ✅ REAL</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box fake'>RIGHT: ❌ FAKE</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='conf-text'>Confidence: {confidence2*100:.2f}%</div>", unsafe_allow_html=True)
            st.progress(float(confidence2))

    st.caption("Threshold = 0.5")