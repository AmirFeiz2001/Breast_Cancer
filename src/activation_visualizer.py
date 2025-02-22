import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.utils import normalize
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_modifier(m):
    #Modify model to use linear activation in the last layer
    m.layers[-1].activation = tf.keras.activations.linear
    return m

def loss(output):
    return output

def plot_saliency(model, X, output_dir="output/plots"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saliency = Saliency(model, model_modifier=model_modifier, clone=False)
    saliency_map = saliency(loss, X)
    saliency_map = normalize(saliency_map)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(saliency_map[0], cmap='jet')
    fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax)
    plt.title("Saliency Map")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "saliency_map.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saliency map saved to {output_path}")

def plot_gradcam(model, X, images, output_dir="output/plots", method="gradcam"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if method == "gradcam":
        gradcam = Gradcam(model, model_modifier=model_modifier, clone=False)
        title = "Grad-CAM"
    elif method == "gradcam++":
        gradcam = GradcamPlusPlus(model, model_modifier=model_modifier, clone=False)
        title = "Grad-CAM++"
    else:
        raise ValueError("Method must be 'gradcam' or 'gradcam++'")

    cam = gradcam(loss, X, penultimate_layer=-1)
    cam = normalize(cam)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), subplot_kw={'xticks': [], 'yticks': []})
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    ax.imshow(images[0], cmap='gray')
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax)
    plt.title(title)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{method}.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"{title} heatmap saved to {output_path}")

def plot_original_image(images, output_dir="output/plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3), subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(images[0], cmap='gray')
    plt.title("Original Image")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "original_image.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Original image saved to {output_path}")
