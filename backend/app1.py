from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask_cors import CORS

CORS(app, supports_credentials=True)
# Load the trained model
model = load_model('face_recognition_model.keras')

# Load label dictionary
label_dict = np.load('label_dict.npy', allow_pickle=True).item()

app = Flask(__name__)

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'message': 'No image part'}), 400
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image for the model
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict the face
    predictions = model.predict(img)
    max_index = np.argmax(predictions)
    person = label_dict[max_index]
    confidence = predictions[0][max_index] * 100
    
    return jsonify({'person': person, 'confidence': confidence})

@app.route('/train_new_face', methods=['POST'])
def train_new_face():
    if 'image' not in request.files:
        return jsonify({'message': 'No image part'}), 400
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image and save it to a directory for training
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Add logic here to save the image for future training or fine-tuning

    return jsonify({'message': 'Face trained successfully!'})


if __name__ == '__main__':
    app.run(debug=True)
