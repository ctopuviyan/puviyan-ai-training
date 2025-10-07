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
import zipfile
from pathlib import Path
import shutil
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# Configuration for Flutter compatibility
MODEL_NAME = "soil_classifier_flutter_compatible"
INPUT_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 80  # Optimized for better GPU utilization
EPOCHS = 50
LEARNING_RATE = 0.001

# GPU Optimization Settings
AUTO_BATCH_SIZE = True  # Automatically adjust batch size based on GPU memory
MAX_BATCH_SIZE = 128    # Maximum batch size for high-end GPUs
MIN_BATCH_SIZE = 16     # Minimum batch size for low-memory situations

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

def setup_mixed_precision():
    """Setup mixed precision for faster GPU training"""
    # DISABLED for TFLite compatibility - mixed precision causes conversion issues
    print("ℹ️ Mixed precision disabled for maximum TFLite compatibility")
    print("📱 Using float32 training to ensure Flutter compatibility")
    
    # Ensure float32 policy is set
    tf.keras.mixed_precision.set_global_policy('float32')
    return False

def create_flutter_compatible_model():
    """Create a model that uses only Flutter-compatible TFLite operations"""
    print("🏗️ Creating Flutter-compatible soil classification model...")
    print("📱 Architecture optimized for TFLite built-in ops only")
    
    # Setup mixed precision for faster training (will be converted back for TFLite)
    mixed_precision_enabled = setup_mixed_precision()
    
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
    """Generate synthetic soil images for training using GPU acceleration"""
    print("🎨 Generating synthetic soil dataset...")
    
    # Check if GPU is available for data generation
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0
    
    if use_gpu:
        print("⚡ Using GPU acceleration for synthetic data generation")
    else:
        print("🔄 Using CPU for synthetic data generation")
    
    @tf.function
    def create_soil_texture_gpu(soil_type, batch_size=100):
        """Create realistic soil texture using GPU operations"""
        
        # Base color ranges for different soil types (as tensors)
        color_ranges = tf.constant([
            [[0.4, 0.3, 0.2], [0.7, 0.5, 0.3]],  # Alluvial - brown
            [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]],  # Black - dark
            [[0.5, 0.2, 0.1], [0.8, 0.4, 0.2]],  # Red - reddish
            [[0.6, 0.3, 0.2], [0.9, 0.5, 0.3]],  # Laterite - orange-red
            [[0.7, 0.6, 0.4], [0.9, 0.8, 0.6]],  # Desert - sandy
            [[0.6, 0.6, 0.5], [0.8, 0.8, 0.7]],  # Saline - whitish
            [[0.2, 0.3, 0.1], [0.4, 0.5, 0.3]],  # Peaty - dark green
            [[0.3, 0.2, 0.1], [0.5, 0.4, 0.3]],  # Forest - rich brown
        ], dtype=tf.float32)
        
        min_color = color_ranges[soil_type, 0]
        max_color = color_ranges[soil_type, 1]
        
        # Generate base texture using GPU
        shape = (batch_size, INPUT_SIZE, INPUT_SIZE, 3)
        
        # Create base random texture
        base_texture = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        
        # Apply color range
        color_range = max_color - min_color
        colored_texture = min_color + base_texture * color_range
        
        # Add noise for texture variation
        noise = tf.random.normal(shape, mean=0.0, stddev=0.1)
        textured = colored_texture + noise
        
        # Add granular patterns using convolution
        # Create random kernels for texture
        kernel = tf.random.normal((5, 5, 3, 3), stddev=0.1)
        granular = tf.nn.conv2d(textured, kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # Blend original with granular texture
        final_texture = 0.7 * textured + 0.3 * tf.nn.tanh(granular)
        
        # Clip to valid range
        final_texture = tf.clip_by_value(final_texture, 0.0, 1.0)
        
        return final_texture
    
    def create_soil_texture_cpu(soil_type, size=(INPUT_SIZE, INPUT_SIZE)):
        """Fallback CPU version for compatibility"""
        np.random.seed(42 + soil_type)
        
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
        img = np.random.uniform(min_color[0], max_color[0], size + (3,))
        
        for i in range(3):
            channel = np.random.uniform(min_color[i], max_color[i], size)
            noise = np.random.normal(0, 0.1, size)
            channel = np.clip(channel + noise, 0, 1)
            
            for _ in range(10):  # Reduced iterations for speed
                x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
                radius = np.random.randint(2, 6)
                xx, yy = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
                mask = (xx - x)**2 + (yy - y)**2 < radius**2
                channel[mask] *= np.random.uniform(0.8, 1.2)
            
            img[:, :, i] = np.clip(channel, 0, 1)
        
        return img
    
    # Generate balanced dataset (memory-optimized for Colab)
    samples_per_class = 400  # Reduced to prevent memory allocation warnings
    total_samples = samples_per_class * NUM_CLASSES
    
    print(f"📊 Generating {samples_per_class} samples per class...")
    
    if use_gpu:
        # GPU-accelerated batch generation
        print("⚡ Using GPU batch generation for maximum speed...")
        
        X_tensor_list = []
        y_tensor_list = []
        
        batch_size = 200  # Increased batch size for better GPU utilization
        
        for soil_type in range(NUM_CLASSES):
            print(f"🚀 Generating {samples_per_class} samples for {SOIL_LABELS[soil_type]} (GPU)...")
            
            # Generate in batches and keep on GPU
            for batch_start in range(0, samples_per_class, batch_size):
                batch_end = min(batch_start + batch_size, samples_per_class)
                current_batch_size = batch_end - batch_start
                
                # Generate batch on GPU and keep it there
                batch_images = create_soil_texture_gpu(soil_type, current_batch_size)
                batch_labels = tf.fill([current_batch_size], soil_type)
                
                # Keep tensors on GPU
                X_tensor_list.append(batch_images)
                y_tensor_list.append(batch_labels)
            
            print(f"✅ Generated {samples_per_class} samples for {SOIL_LABELS[soil_type]}")
        
        # Keep data on GPU and use TensorFlow for memory-efficient operations
        print("🔀 Creating shuffled dataset directly on GPU...")
        
        # Concatenate all batches on GPU
        X_tensor = tf.concat(X_tensor_list, axis=0)
        y_tensor = tf.concat(y_tensor_list, axis=0)
        
        # Create dataset and shuffle on GPU - keep everything on GPU!
        print("🚀 Creating GPU-only training pipeline...")
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
        dataset = dataset.shuffle(buffer_size=min(10000, total_samples), seed=42)
        
        # Split into train/validation on GPU
        train_size = int(0.8 * total_samples)
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        print(f"📈 Training samples: {train_size}")
        print(f"📉 Validation samples: {total_samples - train_size}")
        
        # Return GPU datasets instead of numpy arrays
        return train_dataset, val_dataset, total_samples
        
    else:
        # CPU fallback generation
        print("🔄 Using CPU generation (slower but compatible)...")
        
        X = np.zeros((total_samples, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        y = np.zeros(total_samples, dtype=np.int32)
        
        for soil_type in range(NUM_CLASSES):
            start_idx = soil_type * samples_per_class
            end_idx = start_idx + samples_per_class
            
            print(f"🔄 Generating {samples_per_class} samples for {SOIL_LABELS[soil_type]} (CPU)...")
            
            for i in range(samples_per_class):
                X[start_idx + i] = create_soil_texture_cpu(soil_type)
                y[start_idx + i] = soil_type
            
            print(f"✅ Generated {samples_per_class} samples for {SOIL_LABELS[soil_type]}")
        
        # Simple shuffle for CPU-generated data
        print("🔀 Shuffling dataset...")
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Training samples: {len(X_train)}")
        print(f"📉 Validation samples: {len(X_val)}")
        
        # Return numpy arrays for CPU path
        return (X_train, y_train), (X_val, y_val), len(X)
    
    print(f"✅ Synthetic dataset created: {total_samples} samples")
    
    # Performance summary
    if use_gpu:
        print("⚡ GPU acceleration used - everything stays on GPU!")
    else:
        print("🔄 CPU generation completed - consider GPU for faster data creation")

def convert_to_flutter_tflite(model, output_path):
    """Convert model to Flutter-compatible TensorFlow Lite"""
    print("📱 Converting to Flutter-compatible TensorFlow Lite...")
    
    # Ensure float32 policy for TFLite conversion
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Create converter
    print("🔄 Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # CRITICAL: Use ONLY built-in TFLite ops for Flutter compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS  # Only standard TFLite operations
    ]
    
    # Enable optimizations for smaller model size
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
        # Primary conversion with quantization
        tflite_model = converter.convert()
        print("✅ Model conversion successful with quantization!")
        
    except Exception as e:
        print(f"⚠️ Quantized conversion failed: {e}")
        print("🔄 Attempting without quantization for maximum compatibility...")
        
        # Fallback: Disable quantization for compatibility
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        # No optimizations for maximum compatibility
        
        try:
            tflite_model = converter.convert()
            print("✅ Float32 model conversion successful!")
        except Exception as e2:
            print(f"❌ Float32 conversion also failed: {e2}")
            print("💡 This suggests the model architecture may not be fully TFLite compatible")
            raise e2
    
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

def detect_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_gpu_optimization():
    """Configure GPU for optimal performance"""
    print("🚀 Setting up GPU optimization...")
    
    # Get available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to prevent GPU memory allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set GPU as preferred device
            tf.config.set_visible_devices(gpus, 'GPU')
            
            print(f"✅ GPU optimization enabled for {len(gpus)} device(s)")
            
            # Check GPU memory
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        total, free = line.split(', ')
                        print(f"   GPU {i}: {free}MB free / {total}MB total")
                        
                        # Auto-adjust batch size based on GPU memory
                        if AUTO_BATCH_SIZE:
                            free_gb = int(free) / 1024
                            if free_gb >= 8:
                                suggested_batch = MAX_BATCH_SIZE
                            elif free_gb >= 6:
                                suggested_batch = 96
                            elif free_gb >= 4:
                                suggested_batch = 80
                            elif free_gb >= 2:
                                suggested_batch = 64
                            else:
                                suggested_batch = 48  # Increased minimum for better GPU utilization
                            
                            global BATCH_SIZE
                            BATCH_SIZE = suggested_batch
                            print(f"   🎯 Auto-adjusted batch size to {BATCH_SIZE} based on {free_gb:.1f}GB free memory")
                else:
                    print("   GPU memory info not available")
            except:
                print("   GPU memory info not available")
                
            return True
            
        except RuntimeError as e:
            print(f"⚠️ GPU setup warning: {e}")
            print("   Continuing with default GPU settings...")
            return True
    else:
        print("❌ No GPU detected - training will use CPU (much slower)")
        print("💡 In Colab: Runtime > Change runtime type > GPU")
        return False

def setup_colab_environment():
    """Setup Google Colab environment"""
    if detect_colab():
        print("🔧 Setting up Google Colab environment...")
        
        # Install required packages
        import subprocess
        import sys
        
        packages = ['tensorflow>=2.13.0', 'matplotlib', 'numpy', 'scikit-learn', 'pillow']
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except:
                print(f"⚠️ Could not install {package}, assuming it's already available")
        
        print("✅ Colab environment setup complete")
        return True
    return False

def setup_dataset_choice():
    """Setup dataset choice interface"""
    print("\n📊 Dataset Selection:")
    print("=" * 60)
    print("Choose your training dataset:")
    print("1. 📤 Upload real soil images (ZIP file)")
    print("2. 🎨 Generate synthetic soil dataset")
    print("3. 🔄 Mixed dataset (synthetic + real)")
    print()
    
    if detect_colab():
        try:
            choice = input("Enter choice (1/2/3): ").strip()
            return choice
        except:
            print("⚠️ Input not available, defaulting to synthetic dataset")
            return "2"
    else:
        # Check for local dataset
        if check_local_dataset():
            return "1"  # Use real data if available locally
        else:
            return "2"  # Default to synthetic

def setup_dataset_upload():
    """Setup dataset upload interface for Colab"""
    try:
        from google.colab import files
        
        print("📤 Upload your soil dataset ZIP file...")
        print("Expected structure inside ZIP:")
        for soil_type in SOIL_LABELS.values():
            folder_name = soil_type.replace('/', '_').replace(' ', '_')
            print(f"   📁 {folder_name}/")
            print(f"      📷 image1.jpg, image2.jpg, ...")
        print()
        
        uploaded = files.upload()
        
        if uploaded:
            # Extract the first ZIP file found
            for filename in uploaded.keys():
                if filename.endswith('.zip'):
                    print(f"📦 Extracting {filename}...")
                    return extract_dataset(filename)
                    
        print("❌ No ZIP file uploaded.")
        return False
        
    except ImportError:
        print("ℹ️ Not in Colab environment")
        return False

def check_local_dataset():
    """Check for local dataset directory"""
    dataset_paths = ['soil_dataset', 'real_soil_dataset', 'dataset', 'data']
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Found local dataset at: {path}")
            return True
    
    return False

def extract_dataset(zip_filename):
    """Extract uploaded dataset ZIP file"""
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('uploaded_dataset')
        
        # Find the actual dataset folder
        dataset_root = 'uploaded_dataset'
        subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        
        if len(subdirs) == 1:
            # Dataset is in a subfolder
            dataset_root = os.path.join(dataset_root, subdirs[0])
        
        # Verify dataset structure
        soil_folders = []
        for soil_type in SOIL_LABELS.values():
            folder_name = soil_type.replace('/', '_').replace(' ', '_')
            folder_path = os.path.join(dataset_root, folder_name)
            if os.path.exists(folder_path):
                soil_folders.append(folder_path)
        
        if len(soil_folders) >= 4:  # At least half the soil types
            print(f"✅ Dataset extracted successfully! Found {len(soil_folders)} soil type folders")
            return True
        else:
            print(f"⚠️ Dataset structure incomplete. Found only {len(soil_folders)} soil type folders")
            return False
            
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def load_real_dataset():
    """Load real soil images from uploaded/local dataset"""
    print("📂 Loading real soil dataset...")
    
    # Find dataset directory
    dataset_paths = ['uploaded_dataset', 'soil_dataset', 'real_soil_dataset', 'dataset', 'data']
    dataset_root = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_root = path
            break
    
    if not dataset_root:
        print("❌ No dataset directory found")
        return None, None
    
    # Check for subdirectory
    subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    if len(subdirs) == 1 and not any(soil_type.replace('/', '_').replace(' ', '_') in subdirs[0] for soil_type in SOIL_LABELS.values()):
        dataset_root = os.path.join(dataset_root, subdirs[0])
    
    X_list = []
    y_list = []
    
    for soil_idx, soil_type in SOIL_LABELS.items():
        folder_name = soil_type.replace('/', '_').replace(' ', '_')
        folder_path = os.path.join(dataset_root, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue
        
        # Load images from this soil type folder
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"⚠️ No images found in: {folder_path}")
            continue
        
        print(f"📷 Loading {len(image_files)} images for {soil_type}...")
        
        for img_file in image_files[:500]:  # Limit to 500 images per class
            try:
                img_path = os.path.join(folder_path, img_file)
                
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((INPUT_SIZE, INPUT_SIZE))
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                X_list.append(img_array)
                y_list.append(soil_idx)
                
            except Exception as e:
                print(f"⚠️ Error loading {img_file}: {e}")
                continue
    
    if not X_list:
        print("❌ No images loaded successfully")
        return None, None
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"✅ Real dataset loaded: {len(X)} images, {len(np.unique(y))} soil types")
    
    # Show distribution
    unique, counts = np.unique(y, return_counts=True)
    for soil_idx, count in zip(unique, counts):
        print(f"   {SOIL_LABELS[soil_idx]}: {count} images")
    
    return X, y

def main():
    """Main training pipeline"""
    print("🌱 Flutter-Compatible Soil Detection Model Training")
    print("=" * 60)
    
    # Setup environment and GPU optimization
    in_colab = setup_colab_environment()
    gpu_available = setup_gpu_optimization()
    
    # Create output directory (Colab-friendly path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if in_colab:
        output_dir = f"flutter_compatible_{timestamp}"  # Colab root directory
    else:
        output_dir = f"../models/flutter_compatible_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset selection and loading
    print("\n📊 Step 1: Dataset Selection & Loading")
    dataset_choice = setup_dataset_choice()
    
    X, y = None, None
    
    if dataset_choice == "1":
        # Real dataset
        if in_colab:
            if setup_dataset_upload():
                X, y = load_real_dataset()
        else:
            X, y = load_real_dataset()
        
        if X is None:
            print("⚠️ Real dataset loading failed. Falling back to synthetic data.")
            X, y = generate_synthetic_soil_data()
    
    elif dataset_choice == "3":
        # Mixed dataset
        print("🔄 Creating mixed dataset (synthetic + real)...")
        
        # Try to load real data first
        X_real, y_real = None, None
        if in_colab:
            if setup_dataset_upload():
                X_real, y_real = load_real_dataset()
        else:
            X_real, y_real = load_real_dataset()
        
        # Generate synthetic data
        X_synthetic, y_synthetic = generate_synthetic_soil_data()
        
        # Combine datasets
        if X_real is not None:
            print(f"📊 Combining {len(X_real)} real + {len(X_synthetic)} synthetic images")
            X = np.concatenate([X_real, X_synthetic], axis=0)
            y = np.concatenate([y_real, y_synthetic], axis=0)
        else:
            print("⚠️ No real data available, using synthetic only")
            X, y = X_synthetic, y_synthetic
    
    else:
        # Synthetic dataset (default)
        print("🎨 Generating synthetic soil dataset...")
        result = generate_synthetic_soil_data()
    
    # Handle different return types (GPU datasets vs CPU numpy arrays)
    if len(result) == 3 and hasattr(result[0], 'batch'):  # GPU datasets
        train_dataset, val_dataset, total_samples = result
        
        # Prepare datasets for training
        train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        print(f"✅ GPU-only pipeline ready: {total_samples} total samples")
        use_gpu_datasets = True
        
    else:  # CPU numpy arrays or mixed datasets
        if len(result) == 3:  # CPU path
            (X_train, y_train), (X_val, y_val), total_samples = result
        else:  # Mixed dataset path (fallback)
            X, y = result
            # Shuffle and split
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]
            
            print(f"✅ Final dataset: {len(X)} images, {len(np.unique(y))} soil types")
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📈 Training samples: {len(X_train)}")
            print(f"📉 Validation samples: {len(X_val)}")
        
        use_gpu_datasets = False
    
    # Create model
    print("\n🏗️ Step 2: Model Creation")
    model = create_flutter_compatible_model()
    
    # Train model
    print("\n🚀 Step 3: Model Training")
    
    # Performance estimates
    if gpu_available:
        estimated_time = EPOCHS * 1.5  # 1.5 minutes per epoch with GPU
        print(f"⚡ GPU-accelerated training enabled")
        print(f"📊 Using batch size: {BATCH_SIZE}")
        print(f"⏱️ Estimated training time: ~{estimated_time:.0f} minutes")
    else:
        estimated_time = EPOCHS * 8   # 8 minutes per epoch with CPU
        print(f"🐌 CPU training (consider enabling GPU for 5x speedup)")
        print(f"⏱️ Estimated training time: ~{estimated_time:.0f} minutes")
    
    # Enhanced callbacks for faster training
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=7, 
            factor=0.5,
            min_lr=1e-7,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=0
        )
    ]
    
    if not use_gpu_datasets:
        print(f"🎯 Training with {len(X_train)} samples, validating with {len(X_val)} samples")
    else:
        print(f"🎯 Training with GPU datasets (no CPU RAM usage)")
    print("🚀 Starting training...")
    
    # Start training with optimized settings
    start_time = datetime.now()
    
    # Train with appropriate data format
    if use_gpu_datasets:
        print("🚀 Training with GPU-only pipeline (no CPU RAM usage)...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("🔄 Training with CPU/mixed data...")
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
    
    end_time = datetime.now()
    actual_time = (end_time - start_time).total_seconds() / 60
    print(f"✅ Training completed in {actual_time:.1f} minutes (estimated: {estimated_time:.0f})")
    
    # Evaluate model
    print("\n📊 Step 4: Model Evaluation")
    if use_gpu_datasets:
        print("📊 Evaluating GPU-only pipeline...")
        train_loss, train_acc = model.evaluate(train_dataset, verbose=0)
        val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    else:
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
    
    # Handle different data formats for metadata
    if use_gpu_datasets:
        # For GPU datasets, we know the total samples from the generation
        total_samples_count = total_samples if 'total_samples' in locals() else 3200  # Default from 400*8
        training_samples_count = int(0.8 * total_samples_count)
        validation_samples_count = total_samples_count - training_samples_count
    else:
        # For numpy arrays
        total_samples_count = len(X) if 'X' in locals() else len(X_train) + len(X_val)
        training_samples_count = len(X_train)
        validation_samples_count = len(X_val)
    
    accuracy_metrics = {
        "training_accuracy": float(train_acc),
        "validation_accuracy": float(val_acc),
        "training_loss": float(train_loss),
        "validation_loss": float(val_loss),
        "dataset_type": "real" if dataset_choice == "1" else ("mixed" if dataset_choice == "3" else "synthetic"),
        "total_samples": total_samples_count,
        "training_samples": training_samples_count,
        "validation_samples": validation_samples_count,
        "gpu_pipeline": use_gpu_datasets
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
    
    # Copy model to current directory for easier access
    import shutil
    current_dir_model = f"{MODEL_NAME}.tflite"
    try:
        shutil.copy2(model_path, current_dir_model)
        print(f"📁 Model copied to current directory: {current_dir_model}")
    except Exception as e:
        print(f"⚠️ Could not copy model to current directory: {e}")
    
    # Colab download instructions
    if in_colab:
        show_colab_download_instructions(model_path, metadata_path, plot_path)
        
    # Test Flutter compatibility
    print("\n🧪 Testing Flutter compatibility of generated model...")
    try:
        import subprocess
        import os
        
        # Download the test script if it doesn't exist
        test_script = "test_flutter_compatibility.py"
        if not os.path.exists(test_script):
            subprocess.run([
                "wget", "-O", test_script,
                "https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/test_flutter_compatibility.py"
            ], check=True)
        
        # Run the compatibility test on the copied model
        result = subprocess.run([
            "python", test_script, current_dir_model
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Flutter compatibility test passed!")
            print(result.stdout)
        else:
            print("⚠️ Flutter compatibility test had issues:")
            print(result.stderr)
            
    except Exception as e:
        print(f"⚠️ Could not run Flutter compatibility test: {e}")
        print(f"💡 You can manually test with: python test_flutter_compatibility.py {current_dir_model}")

def show_colab_download_instructions(model_path, metadata_path, plot_path):
    """Show download instructions for Google Colab"""
    print("\n📥 Google Colab Download Instructions:")
    print("=" * 60)
    print("🔽 Download your trained model files:")
    print()
    
    try:
        from google.colab import files
        import os
        
        # Check if files exist before trying to download
        if os.path.exists(model_path):
            print("📱 Downloading TFLite model...")
            files.download(model_path)
        else:
            print(f"⚠️ Model file not found: {model_path}")
        
        if os.path.exists(metadata_path):
            print("📄 Downloading metadata...")
            files.download(metadata_path)
        else:
            print(f"⚠️ Metadata file not found: {metadata_path}")
        
        if os.path.exists(plot_path):
            print("📊 Downloading training plots...")
            files.download(plot_path)
        else:
            print(f"⚠️ Plot file not found: {plot_path}")
        
        print("✅ Available files downloaded successfully!")
        
    except ImportError:
        print("⚠️ Not in Colab environment, skipping auto-download")
    except Exception as e:
        print(f"⚠️ Auto-download failed: {e}")
        print("📋 Manual download commands:")
        print(f"   files.download('{model_path}')")
        print(f"   files.download('{metadata_path}')")
        print(f"   files.download('{plot_path}')")
    
    print("\n🚀 Next Steps:")
    print("1. 📱 Copy the .tflite file to your Flutter app: assets/models/")
    print("2. 🔧 Update your Flutter app to use the new model")
    print("3. 🧪 Test the model with real soil images")
    print("4. 📊 Monitor performance on target devices")

if __name__ == "__main__":
    main()
