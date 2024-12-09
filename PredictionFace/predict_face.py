
from keras._tf_keras.keras.models import load_model
import face_recognition
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict_face(model, image_path):
  result = ""
  image = face_recognition.load_image_file(image_path)
  encoding = face_recognition.face_encodings(image)
  
  if encoding:
    prediction = model.predict(np.array([encoding[0]]))
    is_face = prediction[0][0] > 0.5  # Umbral: si > 0.5, es Persona desaparecida
    result = is_face
  else:
    result = None

  return result

        
def predict_face_model(image_path, model):
  result = predict_face(model, image_path)

  if result is None: # No hay ningun rostro
    return 0
  elif result: # Se encontro el rostro
    return 1
  else:  # Hay rostro pero no es la persona que se busca
    return 2