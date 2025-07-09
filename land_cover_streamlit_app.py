
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("Modelenv.v1.h5")

# Define class labels (order must match model's training!)
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# App title
st.title("🌍 Land Cover Classification from Satellite Images")

# File uploader
uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (assumed size 150x150 from training)
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"**{predicted_class}**")
