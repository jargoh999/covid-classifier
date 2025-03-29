from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# URL of your hosted model
MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/luj-store.appspot.com/o/covid_classifier.h5?alt=media&token=1334a928-5848-44ca-9852-958ec025848e"

# Model file name
MODEL_PATH = "covid_classifier.h5"
model = None  # Global model variable (loaded later)

def download_model():
    """Download the model if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):  # Check if model already exists
        print("Model already exists. Skipping download.")
        return
    
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Model downloaded successfully.")
    else:
        print(f"Failed to download model. Status code: {response.status_code}")

def load_model():
    """Load the TensorFlow model into memory."""
    global model
    if model is None:  # Load model only if not already loaded
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")

# Ensure model is downloaded before the first request
download_model()

# Preprocess the image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB (ensures 3 channels)
    image = image.resize((224, 224))  # Resize to model's expected size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/")
def home():
    return "Hello, Flask is working!"

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # if request.method == "OPTIONS":
    #     # Handle CORS preflight request
    #     response = jsonify({"message": "CORS preflight successful"})
    #     response.headers["Access-Control-Allow-Origin"] = "*"
    #     response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    #     response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    #     return response, 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)

    # Load the model (only on first request)
    load_model()

    # Make prediction
    predictions = model.predict(processed_image).tolist()
    
    response = jsonify({"prediction": predictions})
    response.headers["Access-Control-Allow-Origin"] = "*"  # Allow all origins
    return response

if __name__ == '__main__':
    # Load the model before starting the server
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8000)
