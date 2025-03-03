import streamlit as st
import os
import urllib.request
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# URL del modelo en GitHub
MODEL_URL = "https://github.com/No-Country-simulation/s21-10-t-data/raw/main/brain-tumor-detection-acc-96-4-cnn.h5"
MODEL_PATH = "brain-tumor-detection-acc-96-4-cnn.h5"

# Función para descargar y verificar el modelo
def download_model():
    try:
        st.write("Descargando el modelo, por favor espera...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        
        # Verificar si el archivo se descargó correctamente
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convertir a MB
            st.write(f"Modelo descargado exitosamente. Tamaño: {file_size:.2f} MB")
            if file_size < 1:  # Si el tamaño es menor a 1MB, es probable que esté corrupto
                st.error("Error: El modelo parece estar corrupto o incompleto.")
                os.remove(MODEL_PATH)  # Eliminar el archivo defectuoso
                return False
        else:
            st.error("Error: No se pudo descargar el modelo.")
            return False
        return True
    except Exception as e:
        st.error(f"Error al descargar el modelo: {e}")
        return False

# Verificar y descargar el modelo si no existe o está corrupto
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    success = download_model()
    if not success:
        st.stop()

# Intentar cargar el modelo
try:
    model = load_model(MODEL_PATH, compile=False)
    st.write("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.write("Revisa que el modelo sea un archivo .h5 válido y que no esté dañado.")
    st.stop()

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
