import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Path to your model and test images
model_path = '/UNIVERSITY/PythonDIP/currency_model.keras'
test_images_folder = '/UNIVERSITY/PythonDIP/dataset/real/10_front'

# Load the model
model = load_model(model_path)

# Function to load and preprocess images
def preprocess_image(img_path, img_size=(100, 100)):
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    return 'Real' if class_index == 0 else 'Fake'

# Loop through all images in the folder and predict
def batch_predict():
    for fname in os.listdir(test_images_folder):
        fpath = os.path.join(test_images_folder, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            prediction = predict_image(fpath)
            print(f"Image: {fname} - Prediction: {prediction}")

# Run batch prediction
batch_predict()
