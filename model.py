from tensorflow.keras.models import load_model

try:
    model = load_model("brain-tumor-detection-acc-96-4-cnn.h5", compile=False)
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
