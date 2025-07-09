import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# File ID from Google Drive (REPLACE THIS)
file_id = "1DCEai3csdtLvYm9pBp-gcd0iZvzm-9Ao"
model_path = "Modelenv.v1.h5"

# Download from Google Drive if not already downloaded
if not os.path.exists(model_path):
    with st.spinner("Downloading model... please wait ⏳"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Define class labels
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# UI
st.title("🌍 Land Cover Classification from Satellite Images")
uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"**{predicted_class}**")
