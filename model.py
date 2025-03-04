import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from PIL import Image
import os
import h5py

# Definir la arquitectura del modelo basada en Xception
def create_model():
    base_model = Xception(weights=None, include_top=False, input_shape=(150, 150, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu", name="dense")(x)
    x = Dropout(0.5, name="dropout")(x)
    output = Dense(4, activation="softmax", name="dense_1")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Clases
CLASS_TYPES = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']

st.title("Brain Tumor Detection App")
st.write("Sube una imagen de MRI para detectar si hay un tumor y su tipo.")

# Cargar el modelo desde el archivo subido
uploaded_model = st.file_uploader("Sube el archivo del modelo (.h5)", type=["h5"])

if uploaded_model is not None:
    with open("temp_model.h5", "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Verificar si el archivo contiene un modelo completo o solo pesos
    with h5py.File("temp_model.h5", "r") as file:
        if "model_weights" in file.keys():
            st.warning("El archivo solo contiene pesos, se usará la arquitectura de Xception.")
            model = create_model()
            
            # Depuración: Mostrar capas esperadas en el modelo
            expected_layers = [layer.name for layer in model.layers]
            available_layers = list(file["model_weights"].keys())
            
            st.write("Capas esperadas en el modelo creado:")
            st.write(expected_layers)
            
            st.write("Capas disponibles en los pesos:")
            st.write(available_layers)
            
            # Cargar solo los pesos coincidentes
            try:
                model.load_weights("temp_model.h5", by_name=True, skip_mismatch=True)
                st.success("Pesos cargados parcialmente, ignorando capas faltantes.")
            except Exception as e:
                st.error(f"Error al cargar los pesos: {e}")
                st.stop()
        else:
            try:
                model = load_model("temp_model.h5")
                st.success("Modelo completo cargado correctamente.")
            except Exception as e:
                st.error(f"Error al cargar el modelo: {e}")
                st.stop()
    
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
