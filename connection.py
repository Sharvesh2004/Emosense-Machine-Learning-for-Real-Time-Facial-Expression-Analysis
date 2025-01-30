from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model(r"C:\Users\kaileshwar\Downloads\M5\model_name.h5")

# Define the emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to preprocess image data
def preprocess_image(image_path):
    try:
        # Load image
        image = Image.open(image_path)
        # Resize image to match model input shape
        image = image.resize((48, 48))
        # Convert image to numpy array
        image_array = np.array(image)
        # Normalize pixel values
        image_array = image_array / 255.0
        # Expand dimensions to match model input shape
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        # Handle errors during image preprocessing
        return None, str(e)

@app.route('/', methods=['POST'])
def predict_emotion():
    try:
        # Get image path from request data
        request_data = request.get_json()
        image_path = request_data.get('image_path')

        if not image_path:
            return jsonify({'error': 'Image path not provided'}), 400

        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image data'}), 400

        # Predict emotion
        prediction = model.predict(processed_image)
        # Get predicted emotion label
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        return jsonify({'emotion': predicted_emotion}), 200

    except Exception as e:
        # Handle other exceptions gracefully
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
