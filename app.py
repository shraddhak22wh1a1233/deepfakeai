import streamlit as st
import numpy as np
from PIL import Image

# ---- Robust model loader: prefers tf_keras (Keras3 pathway), falls back to tf.keras (2.x) ----
def _get_loader():
    try:
        from tf_keras.models import load_model  # works when using keras3 + tf-keras
        return load_model, "tf_keras"
    except Exception:
        from tensorflow.keras.models import load_model  # works on TF/Keras 2.15.x
        return load_model, "tf.keras"

load_model, backend = _get_loader()

st.set_page_config(page_title="Real vs Fake Checker", page_icon="🔎", layout="centered")
st.title("Real vs Fake Image Detector")

with st.expander("Environment info", expanded=False):
    st.write(f"Model backend: *{backend}*")
    st.write("If loading fails, pin TensorFlow/Keras versions as shown in the instructions.")

@st.cache_resource(show_spinner=False)
def load_xception_model():
    # compile=False avoids issues with missing custom losses/metrics
    return load_model("xception_model.h5", compile=False)

# Adjust these to match YOUR notebook’s preprocessing
IMG_H, IMG_W = 128, 128   # keep as per your code
RESCALE = 1.0/255.0       # same normalization you used

# Load model (with a friendly error if versions mismatch)
try:
    model = load_xception_model()
except Exception as e:
    st.error(
        "Could not load xception_model.h5. This is usually a TensorFlow/Keras version mismatch "
        "with SeparableConv2D. See the two install options above."
    )
    st.exception(e)
    st.stop()

file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_W, IMG_H))
    arr = np.asarray(img, dtype=np.float32) * RESCALE
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr

if file is not None:
    pil = Image.open(file).convert("RGB")
    st.image(pil, caption="Uploaded Image", use_column_width=True)

    x = preprocess_pil(pil)
    with st.spinner("Running prediction..."):
        y = model.predict(x, verbose=0)
    # Assume model outputs a single sigmoid prob at y[0][0]
    prob = float(y[0][0])
    label = "Real" if prob >= 0.5 else "Fake"

    st.subheader("Result")
    st.markdown(f"*Prediction:* {label}")
    st.markdown(f"*Confidence:* {prob:.4f}")
    st.progress(min(max(prob if label == "Real" else 1.0 - prob, 0.0), 1.0))

    st.caption(
        "Threshold = 0.5. If your training used a different threshold or label order, adjust the logic accordingly.")