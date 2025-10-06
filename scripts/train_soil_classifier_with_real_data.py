#!/usr/bin/env python3
"""
🌱 Puviyan Soil Detection Training with Real Dataset Support
============================================================

Enhanced training script that supports both synthetic and real soil datasets.

Usage in Google Colab:
1. Upload this script
2. Upload your soil dataset (zip file)
3. Run: python train_soil_classifier_with_real_data.py

Dataset Structure Expected:
soil_dataset/
├── Alluvial_Soil/
├── Black_Soil/
├── Red_Soil/
├── Laterite_Soil/
├── Desert_Soil/
├── Saline_Alkaline_Soil/
├── Peaty_Marshy_Soil/
└── Forest_Hill_Soil/

Author: Puviyan AI Team
Version: 3.0.0 (Real Dataset Support)
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
from pathlib import Path
import shutil
from PIL import Image
import random

# Script configuration
SCRIPT_VERSION = "3.0.0"
MODEL_NAME = "soil_classifier_lite"
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Indian soil types (8 classes)
SOIL_TYPES = {
    0: "Alluvial Soil",
    1: "Black Soil", 
    2: "Red Soil",
    3: "Laterite Soil",
    4: "Desert Soil",
    5: "Saline/Alkaline Soil",
    6: "Peaty/Marshy Soil",
    7: "Forest/Hill Soil"
}

# Folder name mappings for real datasets
FOLDER_MAPPINGS = {
    "alluvial": 0, "alluvial_soil": 0,
    "black": 1, "black_soil": 1, "cotton": 1,
    "red": 2, "red_soil": 2, "lateritic": 2,
    "laterite": 3, "laterite_soil": 3,
    "desert": 4, "desert_soil": 4, "sandy": 4, "arid": 4,
    "saline": 5, "alkaline": 5, "saline_alkaline": 5, "salt": 5,
    "peaty": 6, "marshy": 6, "peaty_marshy": 6, "organic": 6,
    "forest": 7, "hill": 7, "forest_hill": 7, "mountain": 7
}

def check_environment():
    """Check if running in Google Colab and system resources"""
    print(f"🌱 Puviyan Soil Detection Training v{SCRIPT_VERSION}")
    print("=" * 60)
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        IN_COLAB = True
    except ImportError:
        print("ℹ️ Running in local environment")
        IN_COLAB = False
    
    # Check TensorFlow and GPU
    print(f"✅ TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("⚠️ No GPU detected - training will be slower")
    
    print("=" * 60)
    return IN_COLAB

def setup_dataset_upload():
    """Setup dataset upload interface for Colab"""
    try:
        from google.colab import files
        
        print("📁 Dataset Upload Options:")
        print("1. Upload a ZIP file containing soil images")
        print("2. Use synthetic dataset (automatic)")
        print()
        
        choice = input("Enter choice (1 for real dataset, 2 for synthetic): ").strip()
        
        if choice == "1":
            print("📤 Please upload your soil dataset ZIP file...")
            print("Expected structure inside ZIP:")
            for i, soil_type in SOIL_TYPES.items():
                print(f"   {soil_type.replace('/', '_').replace(' ', '_')}/")
            print()
            
            uploaded = files.upload()
            
            if uploaded:
                # Extract the first ZIP file found
                for filename in uploaded.keys():
                    if filename.endswith('.zip'):
                        print(f"📦 Extracting {filename}...")
                        extract_dataset(filename)
                        return True
                        
            print("❌ No ZIP file uploaded. Using synthetic dataset.")
            return False
        else:
            print("🎨 Using synthetic dataset generation...")
            return False
            
    except ImportError:
        print("ℹ️ Not in Colab - checking for local dataset...")
        return check_local_dataset()

def check_local_dataset():
    """Check for local dataset directory"""
    dataset_paths = ['soil_dataset', 'real_soil_dataset', 'dataset', 'data']
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Found local dataset at: {path}")
            return True
    
    print("ℹ️ No local dataset found. Using synthetic generation.")
    return False

def extract_dataset(zip_filename):
    """Extract uploaded dataset ZIP file"""
    dataset_dir = "real_soil_dataset"
    
    # Clean existing dataset
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Analyze extracted structure
    analyze_dataset_structure(dataset_dir)

def analyze_dataset_structure(dataset_dir):
    """Analyze and report dataset structure"""
    print(f"📊 Analyzing dataset structure in {dataset_dir}...")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    soil_folders = {}
    
    for root, dirs, files in os.walk(dataset_dir):
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        
        if image_files:
            folder_name = os.path.basename(root).lower()
            soil_folders[root] = {
                'folder_name': folder_name,
                'image_count': len(image_files),
                'images': image_files[:5]  # Sample of first 5 images
            }
    
    print(f"📁 Found {len(soil_folders)} folders with images:")
    for folder_path, info in soil_folders.items():
        print(f"   {info['folder_name']}: {info['image_count']} images")
    
    # Map folders to soil types
    mapped_folders = map_folders_to_soil_types(soil_folders)
    return mapped_folders

def map_folders_to_soil_types(soil_folders):
    """Map dataset folders to soil type classes"""
    mapped = {}
    
    for folder_path, info in soil_folders.items():
        folder_name = info['folder_name']
        
        # Try to match folder name to soil type
        soil_class = None
        for keyword, class_id in FOLDER_MAPPINGS.items():
            if keyword in folder_name:
                soil_class = class_id
                break
        
        if soil_class is not None:
            mapped[folder_path] = {
                'class_id': soil_class,
                'soil_type': SOIL_TYPES[soil_class],
                'image_count': info['image_count']
            }
            print(f"✅ Mapped '{folder_name}' → {SOIL_TYPES[soil_class]} (Class {soil_class})")
        else:
            print(f"⚠️ Could not map '{folder_name}' to any soil type")
    
    return mapped

def load_real_dataset(dataset_dir="real_soil_dataset"):
    """Load real soil images from dataset directory"""
    print("📸 Loading real soil dataset...")
    
    # Analyze dataset structure
    mapped_folders = analyze_dataset_structure(dataset_dir)
    
    if not mapped_folders:
        print("❌ No mapped soil folders found. Using synthetic dataset.")
        return None, None
    
    # Load images and labels
    images = []
    labels = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for folder_path, mapping in mapped_folders.items():
        class_id = mapping['class_id']
        soil_type = mapping['soil_type']
        
        print(f"📂 Loading {soil_type} images...")
        
        folder_images = []
        for filename in os.listdir(folder_path):
            if Path(filename).suffix.lower() in image_extensions:
                img_path = os.path.join(folder_path, filename)
                
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((INPUT_SIZE, INPUT_SIZE))
                    img_array = np.array(img) / 255.0
                    
                    folder_images.append(img_array)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {img_path}: {e}")
                    continue
        
        if folder_images:
            images.extend(folder_images)
            labels.extend([class_id] * len(folder_images))
            print(f"   ✅ Loaded {len(folder_images)} images for {soil_type}")
        else:
            print(f"   ❌ No valid images found for {soil_type}")
    
    if not images:
        print("❌ No images loaded successfully. Using synthetic dataset.")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Real dataset loaded: {X.shape[0]} images, {len(np.unique(y))} classes")
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("📊 Class distribution:")
    for class_id, count in zip(unique, counts):
        print(f"   {SOIL_TYPES[class_id]}: {count} images")
    
    return X, y

def create_synthetic_soil_dataset():
    """Create synthetic soil dataset (fallback)"""
    print("🎨 Creating synthetic soil dataset...")
    
    images = []
    labels = []
    
    samples_per_class = 200
    
    for class_id, soil_type in SOIL_TYPES.items():
        print(f"  Generating {samples_per_class} samples for {soil_type}...")
        
        for _ in range(samples_per_class):
            # Generate synthetic soil image based on soil characteristics
            img = generate_synthetic_soil_image(class_id)
            images.append(img)
            labels.append(class_id)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Synthetic dataset created: {X.shape[0]} images, {len(SOIL_TYPES)} classes")
    return X, y

def generate_synthetic_soil_image(soil_class):
    """Generate a synthetic soil image based on soil type characteristics"""
    img = np.random.rand(INPUT_SIZE, INPUT_SIZE, 3)
    
    # Soil-specific color characteristics
    if soil_class == 0:  # Alluvial Soil - mixed brown/gray
        base_color = [0.6, 0.5, 0.3]
        noise_level = 0.3
    elif soil_class == 1:  # Black Soil - very dark
        base_color = [0.2, 0.2, 0.2]
        noise_level = 0.2
    elif soil_class == 2:  # Red Soil - reddish brown
        base_color = [0.7, 0.3, 0.2]
        noise_level = 0.25
    elif soil_class == 3:  # Laterite Soil - orange-red
        base_color = [0.8, 0.4, 0.2]
        noise_level = 0.3
    elif soil_class == 4:  # Desert Soil - light sandy
        base_color = [0.9, 0.8, 0.6]
        noise_level = 0.2
    elif soil_class == 5:  # Saline/Alkaline - whitish
        base_color = [0.9, 0.9, 0.8]
        noise_level = 0.15
    elif soil_class == 6:  # Peaty/Marshy - dark brown
        base_color = [0.3, 0.25, 0.15]
        noise_level = 0.25
    else:  # Forest/Hill - rich brown
        base_color = [0.5, 0.35, 0.2]
        noise_level = 0.3
    
    # Apply base color with noise
    for i in range(3):
        img[:, :, i] = np.clip(
            base_color[i] + (np.random.rand(INPUT_SIZE, INPUT_SIZE) - 0.5) * noise_level,
            0, 1
        )
    
    # Add texture patterns
    if soil_class in [0, 3, 4]:  # Sandy/granular texture
        texture = np.random.rand(INPUT_SIZE, INPUT_SIZE) * 0.1
        img += texture[:, :, np.newaxis]
    
    return np.clip(img, 0, 1)

def create_model():
    """Create CNN model for soil classification"""
    print("🧠 Creating soil classification model...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        
        # Feature extraction layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(SOIL_TYPES), activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created successfully")
    print(f"📊 Model parameters: {model.count_params():,}")
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the soil classification model"""
    print("🚀 Starting model training...")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
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
    
    print("✅ Training completed!")
    return history

def save_model_and_metadata(model, history):
    """Save trained model and metadata"""
    print("💾 Saving model and metadata...")
    
    # Save TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    model_filename = f"{MODEL_NAME}.tflite"
    with open(model_filename, 'wb') as f:
        f.write(tflite_model)
    
    # Create metadata
    final_train_acc = max(history.history['accuracy'])
    final_val_acc = max(history.history['val_accuracy'])
    
    metadata = {
        "model_name": MODEL_NAME,
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "input_size": INPUT_SIZE,
        "num_classes": len(SOIL_TYPES),
        "class_labels": SOIL_TYPES,
        "accuracy": final_val_acc,
        "training_epochs": len(history.history['accuracy']),
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        "model_file": model_filename,
        "usage_instructions": {
            "preprocessing": "Resize image to 224x224, normalize to [0,1]",
            "output": "8 class probabilities for Indian soil types",
            "confidence_threshold": 0.75
        }
    }
    
    # Save metadata
    info_filename = f"{MODEL_NAME}_info.json"
    with open(info_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved: {model_filename} ({len(tflite_model)/1024:.1f}KB)")
    print(f"✅ Metadata saved: {info_filename}")
    
    return model_filename, info_filename

def plot_training_history(history):
    """Plot and save training history"""
    print("📊 Plotting training history...")
    
    try:
        # Set matplotlib backend for Colab
        import matplotlib
        matplotlib.use('Agg')
        
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
        
        # Save plots
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.savefig('training_history.jpg', dpi=300, bbox_inches='tight')
        
        # Show in Colab
        try:
            plt.show()
        except:
            pass
        
        # Save history as JSON
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print("✅ Training plots saved:")
        print("  📊 ./training_history.png")
        print("  📊 ./training_history.jpg") 
        print("  📄 ./training_history.json")
        
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")

def show_download_instructions():
    """Show download instructions for Colab users"""
    try:
        import google.colab
        from google.colab import files
        
        print("\n" + "="*60)
        print("📥 DOWNLOAD TRAINED MODEL FILES")
        print("="*60)
        print("Your soil detection model is ready! Download these files:")
        print()
        
        # List generated files
        generated_files = []
        for filename in os.listdir('.'):
            if filename.endswith(('.tflite', '.json', '.png', '.jpg')):
                size = os.path.getsize(filename)
                generated_files.append((filename, size))
        
        for filename, size in generated_files:
            size_kb = size / 1024
            print(f"📄 {filename} ({size_kb:.1f}KB)")
        
        print("\n🚀 To download all files, run this in a new cell:")
        print("```python")
        print("from google.colab import files")
        print("import os")
        print()
        print("for file in os.listdir('.'):")
        print("    if file.endswith(('.tflite', '.json', '.png', '.jpg')):")
        print("        files.download(file)")
        print("```")
        print("\n✅ Ready for mobile deployment!")
        
    except ImportError:
        print("ℹ️ Files saved locally - ready for use!")

def main():
    """Main training pipeline"""
    # Check environment
    in_colab = check_environment()
    
    # Setup dataset
    use_real_data = False
    if in_colab:
        use_real_data = setup_dataset_upload()
    else:
        use_real_data = check_local_dataset()
    
    # Load dataset
    if use_real_data:
        X, y = load_real_dataset()
        if X is None:
            print("⚠️ Real dataset loading failed. Using synthetic data.")
            X, y = create_synthetic_soil_dataset()
    else:
        X, y = create_synthetic_soil_dataset()
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    
    # Create and train model
    model = create_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Save model and create plots
    model_file, info_file = save_model_and_metadata(model, history)
    plot_training_history(history)
    
    # Show final results
    final_acc = max(history.history['val_accuracy'])
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final validation accuracy: {final_acc:.1%}")
    print(f"💾 Model saved as: {model_file}")
    print(f"📄 Metadata saved as: {info_file}")
    
    # Show download instructions if in Colab
    if in_colab:
        show_download_instructions()

if __name__ == "__main__":
    main()
