import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Configuración de la aplicación Streamlit
st.title("🧠 Clasificación de Tumores Cerebrales con CNN")
st.write("Sube un modelo preentrenado y una imagen de MRI para predecir si hay un tumor y su tipo.")

# Diccionario de clases
CLASS_LABELS = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# Cargar el modelo desde la PC
st.sidebar.header("📥 Subir Modelo")
uploaded_model = st.sidebar.file_uploader("Sube el modelo (.h5 o carpeta SavedModel)", type=["h5", "zip"])

# Verificar si se subió el modelo
model = None
if uploaded_model:
    try:
        if uploaded_model.name.endswith(".h5"):
            # Guardar el archivo temporalmente y cargarlo
            with open("temp_model.h5", "wb") as f:
                f.write(uploaded_model.getbuffer())
            model = tf.keras.models.load_model("temp_model.h5")
            st.sidebar.success("✅ Modelo cargado correctamente.")
        else:
            st.sidebar.error("❌ Solo se admiten archivos .h5 por ahora.")
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
