# src/3_train_classifier.py

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import config

def build_classifier(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("--- Starting Classifier Training ---")
    
    # Load features
    print(f"Loading features from '{config.FEATURES_CSV}'...")
    df = pd.read_csv(config.FEATURES_CSV)
    
    features = df.drop(columns=['filename', 'label']).values
    labels = df['label'].values

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    num_classes = len(label_encoder.classes_)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
    )

    # Build and train the model
    model = build_classifier(input_dim=X_train.shape[1], num_classes=num_classes)
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, y_val)
    )

    # Save the model and label encoder
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    model.save(config.MODEL_PATH)
    with open(config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel saved to '{config.MODEL_PATH}'")
    print(f"Label encoder saved to '{config.LABEL_ENCODER_PATH}'")

    # Plot and save history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig('../docs/images/training_history.png')
    print("Training history plot saved to 'docs/images/training_history.png'")
    plt.show()

if __name__ == '__main__':
    main()
