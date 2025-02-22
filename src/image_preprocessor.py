import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        # Load image
        img = load_img(image_path, target_size=target_size)
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_array = img_to_array(img)  # Convert to array
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.reshape(1, *target_size, 1)  # Reshape to (1, H, W, 1)
        logging.info(f"Image {image_path} loaded and preprocessed successfully.")
      
        return img_array
      
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
      
        return None
