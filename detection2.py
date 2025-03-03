import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import io
import matplotlib.pyplot as plt
import os

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Clasificación de Tumores Cerebrales")

st.title("🧠 Clasificación de Tumores Cerebrales")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")
st.write(f"🔹 **Versión de TensorFlow:** `{tf.__version__}`")

# =================== CARGAR MODELO ===================
MODEL_H5_PATH = "brain-tumor-detection-acc-96-4-cnn.h5"
MODEL_TF_PATH = "brain-tumor-detection-acc-96-4-cnn"

st.write(f"📥 **Cargando modelo...**")

model = None

try:
    if os.path.exists(MODEL_H5_PATH):
        model = load_model(MODEL_H5_PATH, compile=False)
        st.success("✅ Modelo cargado exitosamente desde archivo H5")
    elif os.path.exists(MODEL_TF_PATH):
        model = tf.keras.models.load_model(MODEL_TF_PATH)
        st.success("✅ Modelo cargado exitosamente desde formato TensorFlow SavedModel")
    else:
        st.error("❌ No se encontró ningún modelo en la carpeta. Verifica que el archivo está correctamente subido.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

# =================== CLASES DEL MODELO ===================
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# =================== SUBIR UNA IMAGEN ===================
uploaded_file = st.file_uploader("📸 **Sube una imagen médica (JPG, PNG)**", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Leer la imagen y convertirla en array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:

        # Mostrar imagen original
        st.image(image, caption="Imagen original", width=400)

        # 🔹 Preprocesamiento para el modelo
        image_resized = cv2.resize(image, (224, 224))  # Ajustar tamaño
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)  # Convertir a 3 canales
        image_array = np.expand_dims(image_rgb, axis=0)  # Expandir dimensiones

        # =================== REALIZAR P
