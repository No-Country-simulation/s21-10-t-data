import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Clases
CLASS_TYPES = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']

st.title("Brain Tumor Detection App")
st.write("Sube una imagen de MRI para detectar si hay un tumor y su tipo.")

# Cargar el modelo desde el archivo subido
uploaded_model = st.file_uploader("Sube el archivo del modelo (.h5)", type=["h5"])

if uploaded_model is not None:
    with open("temp_model.h5", "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    model = load_model("temp_model.h5")  # Cargar modelo completo en vez de solo pesos
    st.success("Modelo cargado exitosamente")

    uploaded_file = st.file_uploader("Sube una imagen de MRI", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        # Preprocesar la imagen
        image = image.resize((150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0  # Normalización
        
        # Hacer la predicción
        prediction = model.predict(image)
        pred_class = CLASS_TYPES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.subheader(f"Predicción: {pred_class}")
        st.write(f"Confianza: {confidence:.2f}%")
