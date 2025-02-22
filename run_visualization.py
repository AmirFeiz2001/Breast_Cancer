import argparse
from src.image_preprocessor import load_and_preprocess_image
from src.activation_visualizer import plot_saliency, plot_gradcam, plot_original_image
from src.model_utils import load_keras_model, check_gpu_availability
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Visualize Activation Maps for a Neural Network Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .h5 model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output plots')
    args = parser.parse_args()

    # Check GPU availability
    check_gpu_availability()

    # Load model
    model = load_keras_model(args.model_path)
    if model is None:
        return

    # Load and preprocess image
    X = load_and_preprocess_image(args.image_path)
    if X is None:
        return
    images = X[0, :, :, 0:1]  # Extract original image for visualization (remove batch dim)

    # Visualize
    plot_original_image(images, args.output_dir + "/plots")
    plot_saliency(model, X, args.output_dir + "/plots")
    plot_gradcam(model, X, images, args.output_dir + "/plots", method="gradcam")
    plot_gradcam(model, X, images, args.output_dir + "/plots", method="gradcam++")

if __name__ == "__main__":
    main()
