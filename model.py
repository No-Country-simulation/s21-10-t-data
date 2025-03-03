import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "brain-tumor-detection-acc-96-4-cnn.h5"

try:
    print("Intentando cargar el modelo...")
    model = load_model(model_path, compile=False)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)
