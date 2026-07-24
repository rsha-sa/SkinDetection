# Skin Lesion Classification, Segmentation, and Detection

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://skindetection.streamlit.app/)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)

## Overview
This project is an end-to-end artificial intelligence system designed to assist in the early detection of skin cancer. Because skin lesions vary widely in color, shape, and texture, manual evaluation can be inconsistent. This unified computer vision pipeline processes dermoscopic images to simultaneously locate the lesion, outline its exact shape, and classify the specific diagnostic category.

You can test the live application here: **[Skin Detection Streamlit App](https://skindetection.streamlit.app/)**

## Key Features & Model Architecture
The system performs three distinct computer vision tasks, rigorously evaluated across 12 different architectures. To maximize diagnostic reliability, the final pipeline utilizes **Weighted Decision Fusion** to aggregate predictions from the top-performing models.

1. **Classification (5 Models):** 
   * Trained deep learning models (ResNet50, VGG19) and classical ML models (ANN, SVM, KNN).
   * **Final Implementation:** Utilizes a weighted fusion of the models, leveraging a fine-tuned ResNet50 alongside handcrafted feature extraction (Gabor, LBP, HSV) to achieve 86.97% accuracy across 7 diagnostic categories.
2. **Segmentation (4 Models):** 
   * Experimented with SegFormer, Attention U-Net, LightWeightUNet, and DeepLabV3+.
   * **Final Implementation:** SegFormer provides pixel-level masking to outline the lesion's precise shape, achieving a 0.9257 Dice score and 0.9597 pixel accuracy.
3. **Detection (3 Models):** 
   * Evaluated YOLOv8, Faster R-CNN, and RetinaNet.
   * **Final Implementation:** YOLOv8 draws bounding boxes to localize the lesion within the image, achieving a 0.8837 Mean IoU.
  
## Technology Stack
* **Deep Learning Frameworks:** PyTorch, Ultralytics (YOLOv8)
* **Classical ML & Data Processing:** scikit-learn, OpenCV, NumPy, Pandas
* **Web Deployment:** Streamlit (Frontend/Live App)
* **Backend API:** Flask, ngrok (for local tunneling)

## Datasets Used
Models were trained and evaluated using a combination of benchmark dermatology datasets to ensure robust generalization:

* **Classification:** Merged Derm7pt and ISIC 9-Classes datasets.
* **Segmentation:** ISIC 2016 Skin Cancer Segmentation dataset and PH2 dataset.
* **Detection:** HAM10000_bbox dataset.

## Contributors
* Rasha Alsabouni
* Manar Abdullah Alharbi
* Sama Salem Salloum
* Shahad Mohammed Ebrahim
* Mshael Waleed Alshwaf

## Acknowledgment
This project was developed as part of the Artificial Intelligence track at King Abdulaziz University (KAU). It reflects the concepts and practical skills gained throughout the program.
