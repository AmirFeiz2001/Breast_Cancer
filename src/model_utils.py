import tensorflow as tf
from keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_keras_model(model_path):

    try:
        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        return None

def check_gpu_availability():

    from tf_keras_vis.utils import num_of_gpus
    _, gpus = num_of_gpus()
    logging.info(f"{gpus} GPUs available")
    return gpus
