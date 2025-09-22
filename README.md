
# HybridVisionClassifier: Iris Recognition with SIFT + ResNet50

![Project Banner](https://i.imgur.com/gK2Jk8v.png)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a sophisticated, hybrid approach to iris recognition by combining classic computer vision features with modern deep learning techniques. It leverages **SIFT** for keypoint detection and **ResNet50** for deep feature extraction, creating a robust feature set for a final classification model.

---

## 📋 Table of Contents
- [Core Concept](#-core-concept)
- [Project Workflow](#-project-workflow)
- [Key Features](#-key-features)
- [Installation Guide](#-installation-guide)
- [How to Use](#-how-to-use)
- [Prediction on a New Image](#-prediction-on-a-new-image)
- [Project Structure](#-project-structure)
- [Contact](#-contact)

---

## 🧠 Core Concept

The strength of this project lies in its **hybrid feature engineering**:

1.  **SIFT (Scale-Invariant Feature Transform):** A powerful algorithm from classic computer vision used to detect stable, localized keypoints (e.g., unique patterns in the iris texture).
2.  **ResNet50:** A state-of-the-art deep convolutional neural network, pre-trained on ImageNet, used to extract high-level, semantic features from image patches centered around the SIFT keypoints.

By combining SIFT's precise localization with ResNet50's rich descriptive power, we create a feature vector that is both robust and highly discriminative.

---

## 🔄 Project Workflow

The project is structured as a clear, step-by-step pipeline, making it easy to understand and reproduce.

`Raw Data` -> `1_prepare_data.py` -> `Train/Test Split` -> `2_extract_features.py` -> `features.csv` -> `3_train_classifier.py` -> `Trained Model`

---

## ✨ Key Features

-   **Hybrid Feature Model:** Unites the strengths of SIFT and ResNet50 for superior feature representation.
-   **End-to-End Pipeline:** Provides a complete, runnable workflow from data preparation to model training and prediction.
-   **Modular & Scalable Code:** The project is broken down into logical scripts (`prepare`, `extract`, `train`), making it easy to maintain and extend.
-   **Efficient Data Handling:** Saves extracted features to a CSV, allowing for rapid experimentation with different classifiers without re-running the heavy feature extraction process.
-   **Inference Script:** Includes a `predict.py` script to demonstrate how to use the trained model on new, unseen images.

---

## ⚙️ Installation Guide

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/HybridVisionClassifier.git
cd HybridVisionClassifier
```

**2. Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

Follow these steps in order to run the full pipeline.

**Step 0: Place Your Data**
-   Unzip and place your `zoomed_in_eyes` dataset folder inside the `data/raw/` directory.

**Step 1: Prepare the Dataset**
-   Run the first script to split your raw data into training and testing sets.
```bash
cd src
python 1_prepare_data.py
```
-   This will create `train` and `test` folders inside `data/processed/`.

**Step 2: Extract Hybrid Features**
-   This is the most time-consuming step. It will iterate through all training images, extract SIFT+ResNet50 features, and save them to a CSV file.
```bash
python 2_extract_features.py
```
-   A file named `features_and_labels.csv` will be created in the `data/` folder.

**Step 3: Train the Classifier**
-   This script loads the features from the CSV and trains the final classification model.
```bash
python 3_train_classifier.py
```
-   The trained model (`iris_hybrid_classifier.h5`) and a label encoder will be saved in a new `models/` directory.

---

## 🔎 Prediction on a New Image

To test the trained model on a single, new image, use the `predict.py` script.

```bash
python predict.py /path/to/your/new_image.jpg
```
The script will output the predicted class for the image.

---

## 📂 Project Structure

```
HybridVisionClassifier/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── 1_prepare_data.py
    ├── 2_extract_features.py
    ├── 3_train_classifier.py
    └── predict.py
```

---

## 📞 Contact

Yahiaoui Raiane
-   **Email:** `ikba.king2015@gmail.com`
-   **LinkedIn:** [linkedin.com/in/yahiaoui-raiane-253911262](https://www.linkedin.com/in/yahiaoui-raiane-253911262/)
