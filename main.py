from fastapi import FastAPI, File, UploadFile
from PIL import Image
import onnxruntime as ort
import numpy as np
import io
import math
app = FastAPI()
import face
# ort_session = ort.InferenceSession('models/binary_classifier_3.onnx')
# TRHESHOLD = 0.5


# @app.get("/2")
# async def root():
#     return {"message": "Hello World"}

@app.post("/verification")
async def verification(ruta):

    output=face.verification_model(ruta)

    return {
      ""+output
    }

@app.post("/prediction")
async def predict(ruta):

    output=face.emotion_re_model(ruta)

    return {
      ""+output
    }