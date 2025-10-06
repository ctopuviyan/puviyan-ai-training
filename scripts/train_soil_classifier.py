#!/usr/bin/env python3
"""
Soil Type Classification Model Training Script
Creates a TensorFlow Lite model for on-device soil detection

Indian Soil Types:
0. Alluvial Soil
1. Black Soil  
2. Red Soil
3. Laterite Soil
4. Desert Soil
5. Saline/Alkaline Soil
6. Peaty/Marshy Soil
7. Forest/Hill Soil
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "soil_classifier_lite"
INPUT_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Indian soil type labels
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

def create_synthetic_dataset():
    """
    Create synthetic soil image dataset for demonstration
    In production, replace with real soil image dataset
    """
    print("🎨 Creating synthetic soil dataset...")
    
    # Generate synthetic data that mimics soil characteristics
    def generate_soil_texture(soil_type, size=(INPUT_SIZE, INPUT_SIZE, 3)):
        """Generate synthetic soil texture based on type"""
        np.random.seed(42 + soil_type)  # Reproducible random generation
        
        if soil_type == 0:  # Clay - darker, smoother
            base_color = np.array([101, 67, 33]) / 255.0  # Dark brown
            noise_scale = 0.1
        elif soil_type == 1:  # Sandy - lighter, granular
            base_color = np.array([194, 154, 108]) / 255.0  # Sandy brown
            noise_scale = 0.3
        elif soil_type == 2:  # Loamy - balanced, rich
            base_color = np.array([139, 69, 19]) / 255.0  # Saddle brown
            noise_scale = 0.2
        elif soil_type == 3:  # Silty - smooth, medium
            base_color = np.array([160, 82, 45]) / 255.0  # Sienna
            noise_scale = 0.15
        else:  # Rocky - gray, rough
            base_color = np.array([105, 105, 105]) / 255.0  # Dim gray
            noise_scale = 0.4
        
        # Create base image
        image = np.full(size, base_color)
        
        # Add texture noise
        noise = np.random.normal(0, noise_scale, size)
        image = np.clip(image + noise, 0, 1)
        
        # Add some realistic variations
        if soil_type == 1:  # Sandy - add granular pattern
            for _ in range(50):
                x, y = np.random.randint(0, INPUT_SIZE, 2)
                radius = np.random.randint(2, 8)
                color_var = np.random.normal(0, 0.1, 3)
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + color_var, 0, 1)
        
        elif soil_type == 4:  # Rocky - add rock-like patches
            for _ in range(20):
                x, y = np.random.randint(10, INPUT_SIZE-10, 2)
                radius = np.random.randint(5, 15)
                rock_color = np.array([0.4, 0.4, 0.4]) + np.random.normal(0, 0.1, 3)
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(rock_color, 0, 1)
        
        return image.astype(np.float32)
    
    # Generate dataset
    samples_per_class = 200
    total_samples = samples_per_class * NUM_CLASSES
    
    X = np.zeros((total_samples, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int32)
    
    for soil_type in range(NUM_CLASSES):
        start_idx = soil_type * samples_per_class
        end_idx = start_idx + samples_per_class
        
        print(f"  Generating {samples_per_class} samples for {SOIL_LABELS[soil_type]}...")
        
        for i in range(samples_per_class):
            X[start_idx + i] = generate_soil_texture(soil_type)
            y[start_idx + i] = soil_type
    
    # Shuffle dataset
    indices = np.random.permutation(total_samples)
    X = X[indices]
    y = y[indices]
    
    # Split into train/validation
    split_idx = int(0.8 * total_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✅ Dataset created: {len(X_train)} training, {len(X_val)} validation samples")
    return (X_train, y_train), (X_val, y_val)

def create_model():
    """Create the soil classification model"""
    print("🏗️ Creating soil classification model...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        
        # Feature extraction layers (MobileNet-inspired for efficiency)
        layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Global average pooling instead of flatten (reduces parameters)
        layers.GlobalAveragePooling2D(),
        
        # Classification head
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created")
    model.summary()
    return model

def train_model(model, train_data, val_data):
    """Train the soil classification model"""
    print("🚀 Starting model training...")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training completed")
    return history

def convert_to_tflite(model, output_path):
    """Convert Keras model to TensorFlow Lite"""
    print("📱 Converting to TensorFlow Lite...")
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for mobile deployment
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use float16 quantization for smaller model size
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model size
    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ TensorFlow Lite model saved: {output_path}")
    print(f"📊 Model size: {model_size_mb:.2f} MB")
    
    return tflite_model

def test_tflite_model(tflite_model, test_data):
    """Test the TensorFlow Lite model"""
    print("🧪 Testing TensorFlow Lite model...")
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test with a few samples
    X_test, y_test = test_data
    correct_predictions = 0
    total_predictions = min(50, len(X_test))  # Test with 50 samples
    
    for i in range(total_predictions):
        # Prepare input
        input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        
        if predicted_class == y_test[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"✅ TensorFlow Lite model accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    return accuracy

def save_model_info(model_path, accuracy, history):
    """Save model information and metadata"""
    print("💾 Saving model information...")
    
    model_info = {
        "model_name": MODEL_NAME,
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "class_labels": SOIL_LABELS,
        "accuracy": float(accuracy),
        "training_epochs": len(history.history['accuracy']),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "model_file": os.path.basename(model_path),
        "usage_instructions": {
            "preprocessing": "Resize image to 224x224, normalize to [0,1]",
            "output": "5 class probabilities for Clay, Sandy, Loamy, Silty, Rocky",
            "confidence_threshold": 0.75
        }
    }
    
    info_path = model_path.replace('.tflite', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Model info saved: {info_path}")

def plot_training_history(history, output_dir):
    """Plot and save training history"""
    print("📊 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training history saved: {plot_path}")

def main():
    """Main training pipeline"""
    print("🌱 Starting Soil Classification Model Training")
    print("=" * 50)
    
    # Create output directory
    output_dir = "../assets/models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    train_data, val_data = create_synthetic_dataset()
    
    # Create and train model
    model = create_model()
    history = train_model(model, train_data, val_data)
    
    # Convert to TensorFlow Lite
    model_path = os.path.join(output_dir, f"{MODEL_NAME}.tflite")
    tflite_model = convert_to_tflite(model, model_path)
    
    # Test TensorFlow Lite model
    accuracy = test_tflite_model(tflite_model, val_data)
    
    # Save model information
    save_model_info(model_path, accuracy, history)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    print("\n🎉 Model training completed successfully!")
    print(f"📱 TensorFlow Lite model: {model_path}")
    print(f"📊 Final accuracy: {accuracy:.2%}")
    print(f"💾 Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")
    print("\n✅ Ready for mobile deployment!")

if __name__ == "__main__":
    main()
