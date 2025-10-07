#!/usr/bin/env python3
"""
Enhanced Soil Detection Model Trainer v5.0
---------------------------------------
Features:
- GPU-accelerated training
- Supports Synthetic/Real/Hybrid datasets
- TFLite export for Flutter
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configuration
CONFIG = {
    'model_name': 'soil_classifier_enhanced_v5',
    'input_size': 224,
    'num_classes': 8,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0005,
    'patience': 15,
    'min_delta': 1e-4,
    'gpu_memory_limit': 1024 * 8  # 8GB GPU memory limit
}

# Indian soil types
SOIL_LABELS = {
    0: "Alluvial Soil",
    1: "Black Soil",
    2: "Red Soil",
    3: "Laterite Soil",
    4: "Desert Soil",
    5: "Saline/Alkaline Soil",
    6: "Peaty/Marshy Soil",
    7: "Forest/Hill Soil"
}

def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️ No GPU detected. Training will be very slow!")
        return False
    
    try:
        # Set memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limit
        if CONFIG.get('gpu_memory_limit'):
            memory_limit = CONFIG['gpu_memory_limit']
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
        
        print(f"✅ GPU Configured: {gpus[0]}")
        return True
    except Exception as e:
        print(f"⚠️ Error configuring GPU: {e}")
        return False

def get_data_augmentation():
    """Create data augmentation pipeline"""
    return keras.Sequential([
        # Spatial transformations
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2, fill_mode='reflect'),
        layers.RandomZoom(0.2, fill_mode='reflect'),
        
        # Color transformations
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.2),
        layers.RandomSaturation(0.2),
        
        # Advanced augmentations
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect'),
        
        # Random erasing (cutout)
        layers.RandomZoom(height_factor=(0.8, 1.0), width_factor=(0.8, 1.0), 
                         fill_mode='constant', fill_value=0.0),
        
        # Normalization (to 0-1 range)
        layers.Rescaling(1./255)
    ], name="data_augmentation")

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic soil images"""
    print(f"🎨 Generating {num_samples} synthetic soil samples...")
    
    # [Previous synthetic data generation code]
    # ...
    
    return X, y  # Return numpy arrays

def load_real_data(data_dir='data/soil_dataset'):
    """Load real soil images"""
    print(f"📂 Loading real soil images from {data_dir}...")
    
    # [Previous real data loading code]
    # ...
    
    return X, y  # Return numpy arrays

def create_model():
    """Create enhanced soil classification model"""
    print("🏗️ Creating enhanced model architecture...")
    
    inputs = layers.Input(shape=(CONFIG['input_size'], CONFIG['input_size'], 3))
    
    # Data augmentation
    x = get_data_augmentation()(inputs)
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # [Previous enhanced model architecture]
    # ...
    
    # Output layer
    outputs = layers.Dense(CONFIG['num_classes'], activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    """Main training function"""
    # Setup GPU
    use_gpu = setup_gpu()
    
    # Select dataset type
    print("\n📊 Select dataset type:")
    print("1. Synthetic data (generated)")
    print("2. Real data (from disk)")
    print("3. Hybrid (synthetic + real)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    # Load data based on choice
    if choice == '1':
        X, y = generate_synthetic_data(num_samples=5000)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    elif choice == '2':
        X, y = load_real_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    elif choice == '3':
        # Load both and combine
        X_syn, y_syn = generate_synthetic_data(num_samples=2500)
        X_real, y_real = load_real_data()
        
        X = np.concatenate([X_syn, X_real], axis=0)
        y = np.concatenate([y_syn, y_real], axis=0)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("❌ Invalid choice. Using synthetic data by default.")
        X_train, X_val, y_train, y_val = generate_synthetic_data(num_samples=5000)
    
    # Create and train model
    model = create_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['patience'],
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            f"{CONFIG['model_name']}.h5",
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    # Train
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(f"{CONFIG['model_name']}.tflite", 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Model saved as {CONFIG['model_name']}.tflite")

if __name__ == "__main__":
    print("🌱 Puviyan Soil Detection - Enhanced Model Training")
    print("=" * 60)
    
    # Enable memory growth for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    train()
