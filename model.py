import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import tempfile

# Cargar el modelo entrenado
MODEL_PATH = "brain-tumor-detection-acc-96-4-cnn.h5"  # Asegúrate de que el modelo esté en esta ruta
model = load_model(MODEL_PATH)

def preprocess_image(image):
    """
    Preprocesar una imagen para la clasificación de tumores cerebrales.
    """
    image = image.convert('RGB')  # Asegurar que la imagen esté en formato RGB
    image = image.resize((224, 224))  # Redimensionar la imagen al tamaño esperado por la CNN
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Añadir dimensión de batch
    image = image / 255.0  # Normalizar los valores de píxeles
    return image

# Interfaz en Streamlit
st.title("🧠 Detección de Tumores Cerebrales en MRI")
st.write("Sube una imagen de resonancia magnética y el modelo clasificará si hay presencia de un tumor.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Preprocesar la imagen
    processed_image = preprocess_image(image)
    
    # Realizar la predicción
    prediction = model.predict(processed_image)[0]
    classes = ["No Tumor", "Tumor"]  # Ajusta según el orden de salida del modelo
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Mostrar resultado
    st.subheader("Resultado de la Predicción:")
    st.write(f"**{predicted_class}** con una confianza del **{confidence:.2f}%**")
