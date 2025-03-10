import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import io
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Detección y Segmentación de Tumores")

st.title("🧠 Detección y Segmentación de Tumores Cerebrales")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")

# =================== SUBIR MODELO ===================
st.sidebar.header("📥 Subir modelo de detección (.h5)")
uploaded_model = st.sidebar.file_uploader("🔹 Cargar modelo en formato HDF5 (.h5)", type=["h5"])

if uploaded_model:
    try:
        # Guardar el modelo subido en un archivo temporal
        model_path = "uploaded_model.h5"
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())

        # Cargar el modelo
        model = load_model(model_path, compile=False)
        st.sidebar.success("✅ Modelo cargado exitosamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()
else:
    st.sidebar.warning("⚠️ **Por favor, sube un modelo antes de continuar.**")
    st.stop()

# =================== SUBIR UNA IMAGEN ===================
st.write("📸 **Sube una imagen médica (JPG, PNG)**")
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer la imagen y convertirla en array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        st.image(image, caption="Imagen original", width=400)

        # 🔹 Preprocesamiento para el modelo
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        image_array = np.expand_dims(image_rgb, axis=0)

        # =================== REALIZAR PREDICCIÓN ===================
        st.write("🔍 **Analizando la imagen...**")
        prediction = model.predict(image_array)
        probability = prediction[0][0]
        threshold = 0.7
        tumor_detected = probability >= threshold
        diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

        # Mostrar resultados de la CNN
        st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
        st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")

        if tumor_detected:
            st.warning("⚠️ **El modelo ha detectado un posible tumor. Segmentando...**")

            # Segmentación de la imagen
            blurred = cv2.GaussianBlur(image, (7, 7), 2)
            _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                tumor_contour = max(contours, key=cv2.contourArea)
                tumor_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 2)

                # 📌 Mostrar segmentación
                st.image(cv2.cvtColor(tumor_image, cv2.COLOR_BGR2RGB), width=400, caption="Segmentación del tumor")
            else:
                st.error("❌ No se detectaron contornos significativos en la imagen.")
        else:
            st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")
else:
    st.warning("⚠️ **Por favor, sube una imagen médica para analizar.**")
