import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Definir el modelo simplificado
def create_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    return model

# Cargar el modelo preentrenado
model = create_model()
model.load_weights("brain-tumor-detection-acc-99-4C-cnn.h5")  # Asegurar que el archivo está en la misma carpeta

# Clases
CLASS_TYPES = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']

st.title("Brain Tumor Detection App")
st.write("Sube una imagen de MRI para detectar si hay un tumor y su tipo.")

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
