import tensorflow as tf

# Cargar el modelo con una versión anterior si tienes acceso
model = tf.keras.models.load_model("brain-tumor-detection-acc-96-4-cnn.h5", compile=False)

# Guardar en formato actualizado
model.save("brain-tumor-detection-acc-96-4-cnn.h5", save_format="h5")
