import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Clasificación de Tumores Cerebrales")

st.title("🧠 Clasificación de Tumores Cerebrales")
st.write(f"📌 **Versión de TensorFlow:** `{tf.__version__}`")

# =================== CARGAR MODELO ===================
st.header("📥 Cargando Modelo...")

MODEL_H5_PATH = "braintumor2.h5"

# Verificar si el archivo existe
if not os.path.isfile(MODEL_H5_PATH):
    st.error(f"❌ No se encontró el archivo `{MODEL_H5_PATH}`. Asegúrate de subirlo.")
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
    st.subheader("📊 Resumen del modelo:")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))
except Exception as e:
    st.warning(f"⚠️ No se pudo mostrar el resumen del modelo. Error: {str(e)}")
