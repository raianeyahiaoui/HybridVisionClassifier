# src/predict.py

import cv2
import numpy as np
import pickle
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import config
from src.2_extract_features import extract_patches # Re-use the function

def predict_single_image(image_path, model, resnet_extractor, label_encoder):
    """Predicts the class for a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray_image, None)
    if not keypoints:
        print("No keypoints detected in the image.")
        return "Unknown"

    patches = extract_patches(rgb_image, keypoints, config.PATCH_SIZE)
    if len(patches) == 0:
        print("Could not extract any valid patches.")
        return "Unknown"
        
    preprocessed_patches = preprocess_input(patches)
    features = resnet_extractor.predict(preprocessed_patches, verbose=0)
    aggregated_features = np.mean(features, axis=0).reshape(1, -1)
    
    prediction = model.predict(aggregated_features)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_class_name

def main():
    parser = argparse.ArgumentParser(description="Predict iris class from a single image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    print("--- Loading models for prediction ---")
    # Load the trained classifier
    classifier_model = load_model(config.MODEL_PATH)
    # Load the label encoder
    with open(config.LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    # Load the ResNet50 feature extractor
    resnet_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    print(f"\n--- Predicting class for {args.image_path} ---")
    predicted_class = predict_single_image(args.image_path, classifier_model, resnet_extractor, label_encoder)
    
    if predicted_class:
        print(f"\n>>> Predicted Class: {predicted_class}")

if __name__ == '__main__':
    main()
