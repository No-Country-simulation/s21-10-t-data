import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Clasificación de Tumores Cerebrales")
st.title("🧠 Clasificación de Tumores Cerebrales")

# =================== CARGAR MODELO DESDE ARCHIVO ===================
model_file = st.file_uploader("📥 **Sube tu modelo en formato .h5**", type=["h5"])

if model_file is not None:
    model_path = "uploaded_model.h5"
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
    
    st.write(f"📥 **Cargando modelo desde {model_file.name}...**")
    try:
        model = load_model(model_path, compile=False)
        st.success("✅ Modelo cargado exitosamente")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()
else:
    st.warning("⚠️ **Por favor, sube un modelo .h5 para continuar.**")
    st.stop()

# =================== SUBIR UNA IMAGEN ===================
uploaded_file = st.file_uploader("📸 **Sube una imagen médica (JPG, PNG)**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer la imagen y convertirla en array
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", width=400)
    
    # Preprocesamiento para el modelo
    image = image.convert('RGB')  # Asegurar que la imagen está en RGB
    image = image.resize((224, 224))  # Redimensionar la imagen al tamaño esperado por el modelo
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0  # Normalizar la imagen
    
    # =================== REALIZAR PREDICCIÓN ===================
    st.write("🔍 **Analizando la imagen...**")
    prediction = model.predict(image_array)[0]
    classes = ["No Tumor", "Tumor"]  # Ajusta según la estructura del modelo
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Mostrar resultado
    st.subheader("📌 **Resultado de la Clasificación:**")
    st.write(f"🧠 **Clase Predicha:** `{predicted_class}`")
    st.write(f"📊 **Confianza:** `{confidence:.2f}%`")
