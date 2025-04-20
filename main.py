from fastapi import FastAPI, File, UploadFile, HTTPException
import io
import numpy as np
import tensorflow as tf
import os
import cv2
from keras.preprocessing import image as i1
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Charger le modèle TensorFlow Lite au démarrage de l'application
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    logger.info("Le modèle TensorFlow Lite a été chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle.")

# Fonction de classification
def predict_label(img_bytes):
    try:
        # Conversion de l'image en tableau NumPy
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.error("L'image est invalide.")
            raise HTTPException(status_code=400, detail="Format d'image invalide")

        # Redimensionnement à 50x50
        resized = cv2.resize(img, (50, 50))  
        img_array = i1.img_to_array(resized) / 255.0  # Normalisation
        img_array = img_array.reshape(1, 50, 50, 3)  # Ajustement de la forme pour le modèle

        # Préparer les tensors pour TensorFlow Lite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Remplir les entrées du modèle TensorFlow Lite
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Exécuter l'inférence
        interpreter.invoke()

        # Obtenir les résultats
        result = interpreter.get_tensor(output_details[0]['index'])

        # Convertir les résultats en probabilités
        a = round(float(result[0, 0]), 2) * 100  # Convertir numpy.float32 en float
        b = round(float(result[0, 1]), 2) * 100  # Convertir numpy.float32 en float
        probability = [a, b]
        threshold = 10

        if a > threshold or b > threshold:
            ind = np.argmax(result)
            classes = ["Cellule Normal: Pas de Paludisme", "Cellule Infectée: Présence du Paludisme"]
            return classes[ind], probability[ind]
        else:
            return "Image invalide", 0
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction.")

# Endpoint pour l'upload et la classification
@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Lecture du fichier en mémoire
        img_bytes = await file.read()
        logger.info(f"Fichier reçu: {file.filename}")

        # Classification
        label, probability = predict_label(img_bytes)
        return {"filename": file.filename, "label": label, "probability": probability}

    except Exception as e:
        logger.error(f"Erreur dans l'endpoint : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Route d'accueil
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de classification du paludisme"}
