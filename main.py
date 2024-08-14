import os
import warnings
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow logs
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged (default behavior), 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

# Suppress specific deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

app = FastAPI()

# Load the pre-trained model
MODEL = tf.keras.models.load_model(r'C:\Users\eshwar reddy\ML_PROJECTS\DeepLearningProjects\potatodiseaseDLP\models\1\1.keras')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get('/ping')
async def ping():
    return 'Hello, I am alive. Welcome!'


def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Preprocess the image
    img_resized = tf.image.resize(image, (256, 256))  # Resize to the input size expected by your model
    img_bt = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Make predictions
    predictions = MODEL.predict(img_bt)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
