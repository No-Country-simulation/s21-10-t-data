import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import os

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Clasificación de Tumores Cerebrales")

st.title("🧠 Clasificación de Tumores Cerebrales")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{tf.__version__}`")

# =================== CARGAR MODELO ===================
st.header("📥 Cargando Modelo...")
MODEL_H5_PATH = "brain-tumor-detection-acc-80-2.h5"

# Verificar si el archivo existe
if not os.path.exists(MODEL_H5_PATH):
    st.error(f"❌ No se encontró el archivo `{MODEL_H5_PATH}`. Sube el modelo antes de ejecutar la aplicación.")
    st.stop()

# Verificar si el archivo es un modelo H5 válido
try:
    with h5py.File(MODEL_H5_PATH, 'r') as f:
        st.write("✅ El archivo es un modelo HDF5 válido.")
except Exception as e:
    st.error(f"❌ El archivo no es un modelo HDF5 válido. Error: {str(e)}")
    st.stop()

# Intentar cargar el modelo
try:
    model = load_model(MODEL_H5_PATH, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()
