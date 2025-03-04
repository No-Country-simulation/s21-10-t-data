import streamlit as st
import tensorflow as tf
import numpy as np
import zipfile
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Configuración de la aplicación Streamlit
st.title("🧠 Clasificación de Tumores Cerebrales con CNN")
st.write("Sube un modelo preentrenado y una imagen de MRI para predecir si hay un tumor y su tipo.")

# Diccionario de clases
CLASS_LABELS = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# Cargar el modelo desde la PC
st.sidebar.header("📥 Subir Modelo")
uploaded_model = st.sidebar.file_uploader("Sube el modelo (.h5 o .zip para SavedModel)", type=["h5", "zip"])

# Verificar si se subió el modelo
model = None

if uploaded_model:
    try:
        if uploaded_model.name.endswith(".h5"):
            with open("temp_model.h5", "wb") as f:
                f.write(uploaded_model.getbuffer())

            # Cargar modelo con custom_objects para evitar errores de BatchNormalization
            model = tf.keras.models.load_model("temp_model.h5", custom_objects={"BatchNormalization": tf.keras.layers.BatchNormalization})
            st.sidebar.success("✅ Modelo .h5 cargado correctamente.")

        elif uploaded_model.name.endswith(".zip"):
            with open("temp_model.zip", "wb") as f:
                f.write(uploaded_model.getbuffer())

            # Extraer el modelo en formato SavedModel
            with zipfile.ZipFile("temp_model.zip", "r") as zip_ref:
                zip_ref.extractall("saved_model_dir")

            model = tf.keras.models.load_model("saved_model_dir")
            st.sidebar.success("✅ Modelo SavedModel cargado correctamente.")

        else:
            st.sidebar.error("❌ Solo se admiten archivos .h5 o .zip para SavedModel.")

    except Exception as e:
        st.sidebar.error(f"⚠️ Error al cargar el modelo: {e}")

# Subir imagen para predicción
st.header("🖼️ Subir Imagen de MRI")
uploaded_file = st.file_uploader("Sube una imagen de MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Imagen Cargada", use_column_width=True)

    # Preprocesamiento de la imagen
    image = image.resize((150, 150))  # Redimensionar
    image = img_to_array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Expandir dimensiones

    # Verificar si el modelo está cargado antes de hacer la predicción
    if model is not None:
        # Hacer la predicción
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Mostrar el resultado
        st.success(f"**🧠 Predicción:** {CLASS_LABELS[class_index]}")
        st.info(f"**📊 Confianza:** {confidence:.2f}")
    else:
        st.warning("⚠️ Sube un modelo antes de hacer una predicción.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Desarrollado por Miguel Ismerio**")
