import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import os

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Clasificación de Tumores Cerebrales")

st.title("🧠 Clasificación de Tumores Cerebrales")
st.write(f"📌 **Versión de TensorFlow:** `{tf.__version__}`")

# =================== CARGAR MODELO ===================
st.header("📥 Cargando Modelo...")

MODEL_H5_PATH = "brain-tumor-detection-acc-80-2.h5"

# Verificar si el archivo existe
if not os.path.isfile(MODEL_H5_PATH):
    st.error(f"❌ No se encontró el archivo `{MODEL_H5_PATH}`. Sube el modelo antes de ejecutar la aplicación.")
    st.stop()

# Verificar si el archivo es un modelo H5 válido
try:
    with h5py.File(MODEL_H5_PATH, 'r') as f:
        if "model_config" not in f.keys() and "keras_version" not in f.attrs:
            raise ValueError("El archivo no parece ser un modelo Keras válido.")
        st.write("✅ El archivo es un modelo HDF5 válido.")
except Exception as e:
    st.error(f"❌ Error al verificar el archivo HDF5. Detalles: {str(e)}")
    st.stop()

# Intentar cargar el modelo
try:
    model = load_model(MODEL_H5_PATH, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo. Detalles: {str(e)}")
    st.stop()

# Mostrar resumen del modelo si se carga correctamente
try:
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))
except Exception as e:
    st.warning(f"⚠️ No se pudo mostrar el resumen del modelo. Error: {str(e)}")
