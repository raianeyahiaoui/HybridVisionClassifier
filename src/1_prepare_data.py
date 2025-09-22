# src/1_prepare_data.py

import os
import shutil
from sklearn.model_selection import train_test_split
import config

def create_train_test_split(base_dir, train_dir, test_dir, test_size=0.2):
    """
    Splits the raw image data into training and testing sets and copies them
    to a new 'processed' directory.
    """
    print("--- Starting Data Preparation ---")
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for class_dir in class_dirs:
        class_path = os.path.join(base_dir, class_dir)
        for side in ['L', 'R']:
            side_dir = os.path.join(class_path, side)
            if os.path.exists(side_dir):
                image_files = [f for f in os.listdir(side_dir) if f.lower().endswith(('.jpg', '.png'))]
                if not image_files: continue

                train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

                # Create destination directories
                train_dest = os.path.join(train_dir, class_dir, side)
                test_dest = os.path.join(test_dir, class_dir, side)
                os.makedirs(train_dest, exist_ok=True)
                os.makedirs(test_dest, exist_ok=True)

                # Copy files
                for f in train_files: shutil.copy(os.path.join(side_dir, f), train_dest)
                for f in test_files: shutil.copy(os.path.join(side_dir, f), test_dest)
                
                print(f"Processed {side_dir}: {len(train_files)} train, {len(test_files)} test images.")
    
    print("--- Data preparation complete. Processed data is in 'data/processed'. ---")

if __name__ == '__main__':
    create_train_test_split(config.BASE_DIR, config.TRAIN_DIR, config.TEST_DIR)
