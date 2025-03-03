import h5py

model_path = "brain-tumor-detection-acc-96-4-cnn.h5"

try:
    with h5py.File(model_path, 'r') as f:
        print("✅ El archivo es un archivo HDF5 válido.")
except Exception as e:
    print("❌ El archivo NO es un modelo HDF5 válido. Error:", e)
