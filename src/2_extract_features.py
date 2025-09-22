# src/2_extract_features.py

import os
import cv2
import numpy as np
import csv
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
import config

def extract_patches(image, keypoints, patch_size):
    patches = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Ensure the patch is within the image boundaries
        half_patch = patch_size // 2
        y_start, y_end = max(0, y - half_patch), min(image.shape[0], y + half_patch)
        x_start, x_end = max(0, x - half_patch), min(image.shape[1], x + half_patch)
        patch = image[y_start:y_end, x_start:x_end]
        if patch.size == 0: continue
        patch = cv2.resize(patch, (patch_size, patch_size))
        patches.append(patch)
    return np.array(patches)

def main():
    print("--- Starting Feature Extraction ---")
    print("Loading pre-trained ResNet50 model...")
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # SIFT detector
    sift = cv2.SIFT_create()

    # Prepare CSV file
    with open(config.FEATURES_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header: filename, label, feature_vector
        header = ['filename', 'label'] + [f'feature_{i}' for i in range(2048)]
        csv_writer.writerow(header)

        class_dirs = sorted([d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))])
        
        # Use tqdm for a progress bar
        for class_dir in tqdm(class_dirs, desc="Processing Classes"):
            label = class_dir
            class_path = os.path.join(config.TRAIN_DIR, class_dir)
            
            for side in ['L', 'R']:
                side_dir = os.path.join(class_path, side)
                if not os.path.exists(side_dir): continue
                
                for image_file in sorted(os.listdir(side_dir)):
                    image_path = os.path.join(side_dir, image_file)
                    image = cv2.imread(image_path)
                    if image is None: continue

                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

                    # 1. SIFT Feature Extraction
                    keypoints, _ = sift.detectAndCompute(gray_image, None)
                    if not keypoints: continue

                    # 2. Patch Extraction around Keypoints
                    patches = extract_patches(rgb_image, keypoints, config.PATCH_SIZE)
                    if len(patches) == 0: continue
                    
                    # 3. ResNet50 Feature Extraction from Patches
                    preprocessed_patches = preprocess_input(patches)
                    features = base_model.predict(preprocessed_patches, verbose=0)
                    
                    # 4. Aggregate features (average pooling)
                    aggregated_features = np.mean(features, axis=0)

                    # Write to CSV
                    row = [image_file, label] + aggregated_features.tolist()
                    csv_writer.writerow(row)

    print(f"--- Feature extraction complete. Features saved to '{config.FEATURES_CSV}'. ---")

if __name__ == '__main__':
    main()
