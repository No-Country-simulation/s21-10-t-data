import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Cargar el modelo entrenado
model = tf.keras.models.load_model("brain_tumor_classifier.h5")

# Diccionario de clases
CLASS_LABELS = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# Configuración de la aplicación Streamlit
st.title("Clasificación de Tumores Cerebrales con CNN")
st.write("Sube una imagen de MRI para detectar si hay un tumor y su tipo.")

# Cargar imagen
uploaded_file = st.file_uploader("Sube una imagen de MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento de la imagen
    image = image.resize((150, 150))  # Redimensionar a 150x150
    image = img_to_array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Expandir dimensiones

    # Hacer la predicción
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Mostrar el resultado
    st.write(f"**Predicción:** {CLASS_LABELS[class_index]}")
    st.write(f"**Confianza:** {confidence:.2f}")
