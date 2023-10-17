# Import standard dependencies
import cv2
import os
import random
import numpy as np

# from deepface import Deepface
import tensorflow as tf
import uuid
from deepface import DeepFace
import glob
from deepface.basemodels import VGGFace

from tensorflow.keras.preprocessing.image import save_img
import os


def preprocess(file_path):

    i = 0
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    if "label" not in file_path:

        nombre = os.path.basename(file_path)

        nombre_ruta = os.path.join("clean", nombre)
        nuevo_n = nombre_ruta.replace(" ", "")

    save_img(nuevo_n, img)
    i = i+1
    # Return image
    return img


directorio = "mydb"


extensiones = ["*.jpg"]


imagenes = []


for extension in extensiones:
    imagenes.extend(glob.glob(os.path.join(directorio, extension)))


for imagen in imagenes:
    preprocess(imagen)


def verification_model(imagen):

    metrics = ["cosine", "euclidean", "euclidean_l2"]

    dfs = DeepFace.find(img_path=imagen,
                        db_path="clean",
                        distance_metric=metrics[2],
                        enforce_detection=False)
   
    
 
    # list[0].name
    return dfs

def emotion_re_model(imagen):
    objs = DeepFace.analyze(img_path=imagen,
                            actions=['emotion']
                            )

    t_emotion = objs[0]['dominant_emotion']

    return t_emotion


print(verification_model("label.jpg"))