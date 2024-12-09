from keras._tf_keras.keras.models import load_model
import face_recognition
import numpy as np
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def predict_face(model, image_path):
  image = face_recognition.load_image_file(image_path)
  encoding = face_recognition.face_encodings(image)
  
  if encoding:
    prediction = model.predict(np.array([encoding[0]]))
    return prediction[0][0] > 0.5  # True si es una coincidencia
  return False


def predict_face_multiple(folder_path, model):
  folder = Path(folder_path)

  # Buscar imágenes con extensiones comunes
  image_files = sorted(
    list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
  )

  # Procesar cada imagen y agrupar por rango de tiempo
  ranges = []
  start_time = None

  for image_path in image_files:
    image_name = image_path.name
    timestamp_str = image_name.split("_")[-1].split(".")[0]  # Extraer `22-21-54` del nombre
    if predict_face(model, str(image_path)):
      if not start_time:
        start_time = timestamp_str
      end_time = timestamp_str
    elif start_time:
      # Crear una tupla de rango cuando se termina el intervalo
      ranges.append((start_time, end_time))
      start_time = None

  # Agregar el último rango si queda sin cerrar
  if start_time:
    ranges.append((start_time, end_time))

  return ranges