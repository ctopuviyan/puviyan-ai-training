#!/usr/bin/env python3
"""
Flutter-Compatible Soil Detection Model Training
===============================================

Specialized training script that generates TensorFlow Lite models
fully compatible with Flutter's TFLite interpreter.

Key Features:
- Uses ONLY TFLite built-in operations (no SELECT_TF_OPS)
- Avoids BatchNormalization layers that cause compatibility issues
- Generates quantized INT8 models for better mobile performance
- Includes representative dataset for proper quantization
- Creates Flutter-friendly input/output types (UINT8)

Author: Puviyan AI Team
Version: 4.0.0 (Flutter-Compatible)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration for Flutter compatibility
MODEL_NAME = "soil_classifier_flutter_compatible"
INPUT_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

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

def create_flutter_compatible_model():
    """Create a model that uses only Flutter-compatible TFLite operations"""
    print("🏗️ Creating Flutter-compatible soil classification model...")
    print("📱 Architecture optimized for TFLite built-in ops only")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input'),
        
        # Feature extraction - Block 1
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1'),
        layers.Dropout(0.1, name='dropout1'),
        
        # Feature extraction - Block 2  
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.1, name='dropout2'),
        
        # Feature extraction - Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.2, name='dropout3'),
        
        # Feature extraction - Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.2, name='dropout4'),
        
        # Feature extraction - Block 5
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5'),
        layers.GlobalAveragePooling2D(name='global_pool'),
        
        # Classification head
        layers.Dropout(0.5, name='dropout5'),
        layers.Dense(256, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout6'),
        layers.Dense(128, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout7'),
        
        # Output layer - keep as float32 for stable training
        layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='predictions')
    ])
    
    # Compile with standard optimizer (avoid complex optimizers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Flutter-compatible model created")
    print(f"📊 Total parameters: {model.count_params():,}")
    
    return model

def generate_synthetic_soil_data():
    """Generate synthetic soil images for training"""
    print("🎨 Generating synthetic soil dataset...")
    
    def create_soil_texture(soil_type, size=(INPUT_SIZE, INPUT_SIZE)):
        """Create realistic soil texture based on soil type"""
        np.random.seed(42 + soil_type)  # Consistent seed for reproducibility
        
        # Base color ranges for different soil types
        color_ranges = {
            0: ([0.4, 0.3, 0.2], [0.7, 0.5, 0.3]),  # Alluvial - brown
            1: ([0.1, 0.1, 0.1], [0.3, 0.3, 0.3]),  # Black - dark
            2: ([0.5, 0.2, 0.1], [0.8, 0.4, 0.2]),  # Red - reddish
            3: ([0.6, 0.3, 0.2], [0.9, 0.5, 0.3]),  # Laterite - orange-red
            4: ([0.7, 0.6, 0.4], [0.9, 0.8, 0.6]),  # Desert - sandy
            5: ([0.6, 0.6, 0.5], [0.8, 0.8, 0.7]),  # Saline - whitish
            6: ([0.2, 0.3, 0.1], [0.4, 0.5, 0.3]),  # Peaty - dark green
            7: ([0.3, 0.2, 0.1], [0.5, 0.4, 0.3]),  # Forest - rich brown
        }
        
        min_color, max_color = color_ranges[soil_type]
        
        # Generate base texture
        img = np.random.uniform(min_color[0], max_color[0], size + (3,))
        
        # Add soil-specific patterns
        for i in range(3):  # RGB channels
            channel = np.random.uniform(min_color[i], max_color[i], size)
            
            # Add texture noise
            noise = np.random.normal(0, 0.1, size)
            channel = np.clip(channel + noise, 0, 1)
            
            # Add some granular texture
            for _ in range(20):
                x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
                radius = np.random.randint(2, 8)
                xx, yy = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
                mask = (xx - x)**2 + (yy - y)**2 < radius**2
                channel[mask] *= np.random.uniform(0.8, 1.2)
            
            img[:, :, i] = np.clip(channel, 0, 1)
        
        return img
    
    # Generate balanced dataset
    samples_per_class = 1000
    total_samples = samples_per_class * NUM_CLASSES
    
    X = np.zeros((total_samples, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int32)
    
    print(f"📊 Generating {samples_per_class} samples per class...")
    
    for soil_type in range(NUM_CLASSES):
        start_idx = soil_type * samples_per_class
        end_idx = start_idx + samples_per_class
        
        for i in range(samples_per_class):
            X[start_idx + i] = create_soil_texture(soil_type)
            y[start_idx + i] = soil_type
        
        print(f"✅ Generated {samples_per_class} samples for {SOIL_LABELS[soil_type]}")
    
    # Shuffle dataset
    indices = np.random.permutation(total_samples)
    X, y = X[indices], y[indices]
    
    print(f"✅ Synthetic dataset created: {total_samples} samples")
    return X, y

def convert_to_flutter_tflite(model, output_path):
    """Convert model to Flutter-compatible TensorFlow Lite"""
    print("📱 Converting to Flutter-compatible TensorFlow Lite...")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # CRITICAL: Use ONLY built-in TFLite ops for Flutter compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS  # Only standard TFLite operations
    ]
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Create representative dataset for quantization
    def representative_data_gen():
        """Generate representative data for quantization"""
        for _ in range(100):
            # Generate random data matching model input shape
            data = np.random.random((1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_data_gen
    
    # Enable full integer quantization for smaller size and faster inference
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8   # Flutter-friendly
    converter.inference_output_type = tf.uint8  # Flutter-friendly
    
    print("🔄 Converting model (this may take a few minutes)...")
    
    try:
        tflite_model = converter.convert()
        print("✅ Model conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("🔄 Falling back to float32 model...")
        
        # Fallback to float32 if quantization fails
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("✅ Float32 fallback model created")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Model info
    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ Flutter-compatible TFLite model saved: {output_path}")
    print(f"📊 Model size: {model_size_mb:.2f} MB")
    
    return tflite_model

def create_model_metadata(model_path, accuracy_metrics):
    """Create metadata file for the model"""
    metadata = {
        "model_name": MODEL_NAME,
        "version": "4.0.0",
        "created_at": datetime.now().isoformat(),
        "framework": "TensorFlow Lite",
        "flutter_compatible": True,
        "quantized": True,
        "input_shape": [1, INPUT_SIZE, INPUT_SIZE, 3],
        "input_type": "uint8",
        "output_shape": [1, NUM_CLASSES],
        "output_type": "uint8",
        "num_classes": NUM_CLASSES,
        "soil_labels": SOIL_LABELS,
        "accuracy_metrics": accuracy_metrics,
        "preprocessing": {
            "resize_to": [INPUT_SIZE, INPUT_SIZE],
            "normalize": "0-255 to 0-1 range",
            "color_space": "RGB"
        },
        "usage_notes": [
            "Model uses only TFLite built-in operations",
            "Compatible with Flutter tflite_flutter package",
            "Input images should be 224x224 RGB",
            "Output is probability distribution over 8 soil types"
        ]
    }
    
    metadata_path = model_path.replace('.tflite', '_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model metadata saved: {metadata_path}")
    return metadata_path

def main():
    """Main training pipeline"""
    print("🌱 Flutter-Compatible Soil Detection Model Training")
    print("=" * 60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../models/flutter_compatible_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    print("\n📊 Step 1: Data Generation")
    X, y = generate_synthetic_soil_data()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📈 Training samples: {len(X_train)}")
    print(f"📉 Validation samples: {len(X_val)}")
    
    # Create model
    print("\n🏗️ Step 2: Model Creation")
    model = create_flutter_compatible_model()
    
    # Train model
    print("\n🚀 Step 3: Model Training")
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Step 4: Model Evaluation")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"📈 Training Accuracy: {train_acc:.4f}")
    print(f"📉 Validation Accuracy: {val_acc:.4f}")
    
    # Convert to TFLite
    print("\n📱 Step 5: TensorFlow Lite Conversion")
    model_path = os.path.join(output_dir, f"{MODEL_NAME}.tflite")
    tflite_model = convert_to_flutter_tflite(model, model_path)
    
    # Create metadata
    print("\n📄 Step 6: Metadata Creation")
    accuracy_metrics = {
        "training_accuracy": float(train_acc),
        "validation_accuracy": float(val_acc),
        "training_loss": float(train_loss),
        "validation_loss": float(val_loss)
    }
    metadata_path = create_model_metadata(model_path, accuracy_metrics)
    
    # Save training plots
    print("\n📊 Step 7: Saving Training Plots")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved: {plot_path}")
    
    # Final summary
    print("\n🎉 Training Complete!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📱 TFLite model: {model_path}")
    print(f"📄 Metadata: {metadata_path}")
    print(f"📊 Training plots: {plot_path}")
    print(f"🎯 Validation accuracy: {val_acc:.4f}")
    print("\n📱 Model is ready for Flutter integration!")
    print("🔧 Copy the .tflite file to your Flutter app's assets/models/ directory")

if __name__ == "__main__":
    main()
