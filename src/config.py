# src/config.py

# --- DIRECTORY PATHS ---
# Base directory for the raw dataset
BASE_DIR = '../data/raw/zoomed_in_eyes'

# Directories for the processed (split) data
TRAIN_DIR = '../data/processed/train'
TEST_DIR = '../data/processed/test'

# --- FILE PATHS ---
# Path to save the extracted features and labels
FEATURES_CSV = '../data/features_and_labels.csv'

# Path to save the trained model
MODEL_PATH = '../models/iris_hybrid_classifier.h5'

# Path to save the label encoder
LABEL_ENCODER_PATH = '../models/label_encoder.pkl'

# --- FEATURE EXTRACTION PARAMETERS ---
# Size of the patches to extract around SIFT keypoints
PATCH_SIZE = 224

# --- MODEL TRAINING PARAMETERS ---
# Batch size for training the final classifier
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
