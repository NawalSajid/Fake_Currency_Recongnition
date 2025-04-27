from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('currency_model.keras')

# Function to predict a single image
def predict_image(image_path):
    img_size = (100, 100)  # Same size used during training
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch dimension
        
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]

        if class_idx == 0:
            print(f"Result: REAL currency detected ✅")
        else:
            print(f"Result: FAKE currency detected ❌")
    except Exception as e:
        print(f"Error processing image: {e}")

# Example usage
image_path = input("Enter the path of the currency image you want to test: ")
predict_image(image_path)
