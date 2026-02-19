import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page config
st.set_page_config(page_title="Smart Sorting - Fruit & Vegetable Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f0f9f4;
    }

    .main-box {
        background-color: #ffffff;
        padding: 35px 40px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-top: 30px;
    }

    .title-style {
        color: #00796b;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .subtitle-style {
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        color: #004d40;
        margin-bottom: 45px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .usage-text {
        font-size: 20px;
        font-weight: 500;
        max-width: 800px;
        margin: 0 auto 40px auto;
        line-height: 1.6;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .section-heading {
        color: #00796b;
        font-size: 26px;
        font-weight: 800;
        border-left: 6px solid #4db6ac;
        padding-left: 15px;
        margin-top: 50px;
        margin-bottom: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .upload-box {
        background-color: #e0f2f1;
        padding: 25px 30px;
        border: 3px dashed #26a69a;
        border-radius: 15px;
        margin-top: 25px;
        transition: background-color 0.3s ease;
    }
    .upload-box:hover {
        background-color: #b2dfdb;
    }

    .result-box {
        background-color: #e8f5e9;
        padding: 20px 25px;
        border-radius: 15px;
        box-shadow: 0 3px 10px rgba(0, 150, 136, 0.2);
        margin-top: 30px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .healthy-badge {
        color: #2e7d32;
        background-color: #a5d6a7;
        font-weight: 700;
        padding: 6px 15px;
        border-radius: 30px;
        display: inline-block;
        font-size: 18px;
        margin-top: 10px;
    }
    .rotten-badge {
        color: #c62828;
        background-color: #ef9a9a;
        font-weight: 700;
        padding: 6px 15px;
        border-radius: 30px;
        display: inline-block;
        font-size: 18px;
        margin-top: 10px;
    }

    .info-text {
        font-size: 20px;
        font-weight: 600;
        margin-top: 8px;
    }

    .footer {
        margin-top: 60px;
        text-align: center;
        font-size: 15px;
        color: #555;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title-style">🍎 Smart Sorting AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Classify fruits and vegetables as healthy or rotten using deep learning 🍌🥕</div>', unsafe_allow_html=True)

# Usage instructions + tools used
st.markdown('''
<div class="usage-text">
<ul>
  <li>📉 <strong>Prevents food waste</strong></li>
  <li>🛒 <strong>Helps vendors sort produce</strong></li>
  <li>🏡 <strong>Assists families in choosing fresh food</strong></li>
</ul>

<p>Upload a clear image of any fruit or vegetable using the box below. The AI model will analyze the image and tell you whether it is <strong>healthy or rotten</strong>, along with the confidence score.</p>

<p>This tool supports smarter food decisions and reduces waste using the power of deep learning.</p>

<p><strong>🔧 Tools Used:</strong> TensorFlow | MobileNetV2 | Streamlit</p>
</div>
''', unsafe_allow_html=True)

# Classes
CLASS_NAMES = [
    'Apple_Healthy', 'Apple_Rotten',
    'Banana_Healthy', 'Banana_Rotten',
    'Bellpepper_Healthy', 'Bellpepper_Rotten',
    'Carrot_Healthy', 'Carrot_Rotten',
    'Cucumber_Healthy', 'Cucumber_Rotten',
    'Grape_Healthy', 'Grape_Rotten',
    'Guava_Healthy', 'Guava_Rotten',
    'Jujube_Healthy', 'Jujube_Rotten',
    'Mango_Healthy', 'Mango_Rotten',
    'Orange_Healthy', 'Orange_Rotten',
    'Pomegranate_Healthy', 'Pomegranate_Rotten',
    'Potato_Healthy', 'Potato_Rotten',
    'Strawberry_Healthy', 'Strawberry_Rotten',
    'Tomato_Healthy', 'Tomato_Rotten'
]


# Load model
@st.cache_resource
def load_model_from_file():
    try:
        return load_model("Project files/fruitveg_model.h5", compile=False)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model_from_file()

# Preprocessing
def preprocess_image(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# Prediction
def predict_image(uploaded_file):
    processed = preprocess_image(uploaded_file)
    if processed is None:
        return None
    predictions = model.predict(processed)
    idx = np.argmax(predictions[0])
    confidence = float(predictions[0][idx]) * 100
    label = CLASS_NAMES[idx]
    produce_type, condition = label.split('_')
    return {
        'class': label,
        'produce_type': produce_type,
        'condition': condition,
        'confidence': round(confidence, 2),
        'is_healthy': condition.lower() == 'healthy'
    }

# Upload section
st.markdown('<div class="section-heading">📷 Upload Image</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image of a fruit or vegetable (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction output
if uploaded_file is not None:
    st.image(uploaded_file, caption='📸 Uploaded Image', width=700)  # or any width you like

    if model:
        result = predict_image(uploaded_file)
        if result:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#00796b; font-weight:900;'>🧠 Prediction: {result['class']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-text'>🥗 <b>Produce:</b> {result['produce_type'].capitalize()}</div>", unsafe_allow_html=True)
            if result['is_healthy']:
                st.markdown("<div class='healthy-badge'>✅ Healthy</div>", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("<div class='rotten-badge'>❌ Rotten</div>", unsafe_allow_html=True)
                st.error("😢 This item appears to be rotten.")
            st.markdown(f"<div class='info-text'>📊 <b>Confidence:</b> {result['confidence']}%</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠ Could not process the image.")
    else:
        st.error("🚫 Model not loaded. Please check the file path.")

# About us
st.markdown('<div class="section-heading">👩‍💻 About Us</div>', unsafe_allow_html=True)
st.markdown("""
We are a team of enthusiastic computer science students passionate about AI and sustainability.

This project was built during a tech innovation challenge to reduce food waste and promote better food quality decisions.
""")

# Footer
st.markdown('<div class="footer">© 2025 Smart Sorting Project | Made with ❤️ by Team Innovators</div>', unsafe_allow_html=True)
