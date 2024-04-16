from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from mangum import Mangum
from PIL import Image
import uvicorn
import tensorflow as tf
from io import BytesIO
import numpy as np

app = FastAPI()
handler = Mangum(app)

app.mount("/static", StaticFiles(directory="C:/Users/hasan/OneDrive/Desktop/projects/ML_practise/potato_disease_detection/site/static", html=True), name="static")

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http:127.0.0.1",
    "http://localhost:8080",
]

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("classifer\plant_disease_classifier_91.h5", compile=False)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

class_names = [
 'Bell Pepper Bacterial spot','Healthy Bell Pepper',
 'Potato: Early blight',
 'Potato: Late blight',
 'Healthy Potato',
 'Tomato Bacterial spot',
 'Tomato: Early blight',
 'Tomato: Late blight',
 'Tomato Leaf Mold',
 'Tomato Septoria leaf spot',
 'Tomato: Two spotted spider mite',
 'Tomato Target Spot',
 'Tomato Yellow Leaf Curl Virus',
 'Tomato mosaic virus',
 'Healthy Tomato'
 ]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/")
async def index():
    return FileResponse("C:/Users/hasan/OneDrive/Desktop/projects/ML_practise/potato_disease_detection/site/index.html")
    


@app.post("/predict")
async def predict (
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        "class": predicted_class,
        "confidence": f"{confidence * 100}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
