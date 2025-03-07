# Breast Cancer

| Sample | Saliency Map | Grad-CAM | Grad-CAM ++ |
|---------|---------|---------|---------|
| ![Alt text](results/sample.png) | ![Alt text](results/Saliency.png) | ![Alt text](results/Gradcam.png) | ![Alt text](results/GradCam++.png) |


The Breast Cancer project focuses on visualizing and interpreting deep learning models applied to breast cancer diagnosis. It utilizes techniques like Saliency Maps, Grad-CAM, and Grad-CAM++ to highlight regions in medical images that influence model predictions.



## About the Dataset
- 1st column: MIAS database reference number.

- 2nd column: Character of background tissue: F Fatty ,G Fatty-glandular ,D Dense-glandular

- 3rd column: Class of abnormality present: CALC Calcification ,CIRC Well-defined/circumscribed masses ,SPIC Spiculated masses ,MISC Other, ill-defined masses ,ARCH Architectural distortion ,ASYM Asymmetry ,NORM Normal

- 4th column: Severity of abnormality; B Benign ,M Malignant

- 5th, 6th columns: x,y image-coordinates of centre of abnormality.

- 7th column: Approximate radius (in pixels) of a circle enclosing the abnormality.



## The Structure of Malignant and Benign Tumors
- Benign masses typically have a circular or oval shape, whereas malignant masses are irregular and contain small, needle-like structures.

<p align="center"> <img src="Description/structure.png" alt="Overall Structure" width="600"/> </p>

## Comparison of Benign vs. Malignant Structures
<p align="center"> <img src="Description/structure_benign.png" alt="Benign Structure" width="45%"/> <img src="Description/structure_malignant.png" alt="Malignant Structure" width="45%"/> </p>
