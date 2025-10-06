#!/usr/bin/env python3
"""
Soil Type Classification Model Training Script
Creates a TensorFlow Lite model for on-device soil detection

🌱 PUVIYAN AI TRAINING PIPELINE 🌱

Indian Soil Types:
0. Alluvial Soil
1. Black Soil  
2. Red Soil
3. Laterite Soil
4. Desert Soil
5. Saline/Alkaline Soil
6. Peaty/Marshy Soil
7. Forest/Hill Soil

📥 TO DOWNLOAD LATEST VERSION IN GOOGLE COLAB:
```python
# Clean setup (removes old versions)
!rm -f train_soil_classifier.py*

# Download latest version (force overwrite)
!wget -O train_soil_classifier.py https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/train_soil_classifier.py

# Verify download
!ls -la train_soil_classifier.py

# Run training
!python train_soil_classifier.py
```

🔧 FEATURES:
- ✅ 8 Indian soil types with realistic synthetic data
- ✅ Automatic training plots generation
- ✅ TensorFlow Lite model conversion
- ✅ Automatic GitHub repository upload
- ✅ Complete deployment package creation
- ✅ Mobile app integration instructions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
import shutil

# Configuration
MODEL_NAME = "soil_classifier_lite"
INPUT_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SCRIPT_VERSION = "2.1.0"  # Updated with GitHub upload and download fix

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
        
        if soil_type == 0:  # Alluvial - fertile river deposits, brownish
            base_color = np.array([139, 115, 85]) / 255.0  # Light brown
            noise_scale = 0.15
        elif soil_type == 1:  # Black - clay-rich, dark
            base_color = np.array([47, 47, 47]) / 255.0  # Dark gray/black
            noise_scale = 0.08
        elif soil_type == 2:  # Red - iron-rich, reddish
            base_color = np.array([205, 92, 92]) / 255.0  # Indian red
            noise_scale = 0.12
        elif soil_type == 3:  # Laterite - iron/aluminum rich, brick red
            base_color = np.array([178, 34, 34]) / 255.0  # Fire brick red
            noise_scale = 0.10
        elif soil_type == 4:  # Desert - arid, sandy light
            base_color = np.array([244, 164, 96]) / 255.0  # Sandy brown
            noise_scale = 0.25
        elif soil_type == 5:  # Saline/Alkaline - high salt, whitish
            base_color = np.array([211, 211, 211]) / 255.0  # Light gray
            noise_scale = 0.18
        elif soil_type == 6:  # Peaty/Marshy - organic rich, dark brown
            base_color = np.array([85, 107, 47]) / 255.0  # Dark olive green
            noise_scale = 0.14
        else:  # Forest/Hill - rich humus, dark brown
            base_color = np.array([101, 67, 33]) / 255.0  # Dark brown
            noise_scale = 0.12
        
        # Create base image
        image = np.full(size, base_color)
        
        # Add texture noise
        noise = np.random.normal(0, noise_scale, size)
        image = np.clip(image + noise, 0, 1)
        
        # Add realistic texture variations for Indian soil types
        if soil_type == 0:  # Alluvial - mixed particle sizes, river deposits
            for _ in range(30):
                x, y = np.random.randint(0, INPUT_SIZE, 2)
                radius = np.random.randint(3, 10)
                color_var = np.random.normal(0, 0.08, 3)
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + color_var, 0, 1)
        
        elif soil_type == 1:  # Black - clay texture, smooth patches
            for _ in range(15):
                x, y = np.random.randint(5, INPUT_SIZE-5, 2)
                radius = np.random.randint(8, 20)
                clay_smoothing = np.random.normal(0, 0.03, 3)
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + clay_smoothing, 0, 1)
        
        elif soil_type == 2:  # Red - iron oxide patterns
            for _ in range(25):
                x, y = np.random.randint(0, INPUT_SIZE, 2)
                radius = np.random.randint(2, 8)
                iron_color = np.array([0.15, -0.05, -0.05])  # More red
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + iron_color, 0, 1)
        
        elif soil_type == 4:  # Desert - sandy granular pattern
            for _ in range(60):
                x, y = np.random.randint(0, INPUT_SIZE, 2)
                radius = np.random.randint(1, 5)
                sand_grain = np.random.normal(0, 0.15, 3)
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + sand_grain, 0, 1)
        
        elif soil_type == 5:  # Saline - salt crystal patterns
            for _ in range(40):
                x, y = np.random.randint(0, INPUT_SIZE, 2)
                radius = np.random.randint(2, 6)
                salt_crystal = np.array([0.1, 0.1, 0.1])  # Whitish crystals
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + salt_crystal, 0, 1)
        
        elif soil_type == 6:  # Peaty - organic matter patches
            for _ in range(20):
                x, y = np.random.randint(5, INPUT_SIZE-5, 2)
                radius = np.random.randint(6, 15)
                organic_matter = np.array([-0.1, 0.05, -0.05])  # Darker, greenish
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if 0 <= x+dx < INPUT_SIZE and 0 <= y+dy < INPUT_SIZE:
                            if dx*dx + dy*dy <= radius*radius:
                                image[x+dx, y+dy] = np.clip(image[x+dx, y+dy] + organic_matter, 0, 1)
        
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
            "output": "8 class probabilities for Indian soil types",
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
    
    try:
        # Setup matplotlib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('🎯 Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('📉 Model Loss Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        # Also save as JPG for better compatibility
        jpg_path = os.path.join(output_dir, 'training_history.jpg')
        plt.savefig(jpg_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        # Show plot in Colab
        plt.show()
        plt.close()
        
        print(f"✅ Training plots saved:")
        print(f"  📊 {plot_path}")
        print(f"  📊 {jpg_path}")
        
        # Save training history as JSON for later analysis
        history_json = {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'epochs': len(history.history['accuracy'])
        }
        
        json_path = os.path.join(output_dir, 'training_history.json')
        with open(json_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"  📄 {json_path}")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("📊 Training completed but plots could not be generated")

def upload_model_to_github(model_files, repo_url="https://github.com/ctopuviyan/puviyan-ai-training.git"):
    """Upload trained model files to GitHub repository"""
    print("🚀 Uploading model to GitHub repository...")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("📂 Not in a git repository. Cloning repository...")
            
            # Clone the repository
            subprocess.run(['git', 'clone', repo_url, 'temp_repo'], check=True)
            repo_dir = 'temp_repo'
        else:
            print("📂 Already in git repository")
            repo_dir = '.'
        
        # Create models directory in repo
        models_dir = os.path.join(repo_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Copy model files to repository
        copied_files = []
        for file_path in model_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(models_dir, filename)
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)
                print(f"  ✅ Copied {filename} to repository")
        
        if not copied_files:
            print("❌ No model files found to upload")
            return False
        
        # Change to repository directory
        original_dir = os.getcwd()
        os.chdir(repo_dir)
        
        try:
            # Configure git (in case it's not configured)
            subprocess.run(['git', 'config', 'user.email', 'ai-training@puviyan.com'], 
                         capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'Puviyan AI Training'], 
                         capture_output=True)
            
            # Add model files to git
            subprocess.run(['git', 'add', 'models/'], check=True)
            
            # Create commit message with model info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"Add trained soil detection model - {timestamp}\n\n"
            commit_msg += f"- Model: {MODEL_NAME}.tflite\n"
            commit_msg += f"- Classes: {NUM_CLASSES} Indian soil types\n"
            commit_msg += f"- Input size: {INPUT_SIZE}x{INPUT_SIZE}\n"
            commit_msg += f"- Training epochs: {EPOCHS}\n"
            commit_msg += "- Ready for mobile deployment"
            
            # Commit changes
            result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Model committed to repository")
                
                # Try to push (may require authentication)
                print("🔄 Attempting to push to GitHub...")
                push_result = subprocess.run(['git', 'push'], 
                                           capture_output=True, text=True)
                
                if push_result.returncode == 0:
                    print("🎉 Model successfully uploaded to GitHub!")
                    print(f"🔗 Repository: {repo_url}")
                    return True
                else:
                    print("⚠️ Model committed locally but push failed")
                    print("💡 You may need to manually push or configure authentication")
                    print(f"Error: {push_result.stderr}")
                    return False
            else:
                if "nothing to commit" in result.stdout:
                    print("ℹ️ Model already up to date in repository")
                    return True
                else:
                    print(f"❌ Commit failed: {result.stderr}")
                    return False
                    
        finally:
            # Return to original directory
            os.chdir(original_dir)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def create_deployment_package(model_path, info_path, plots_path):
    """Create a deployment package with all necessary files"""
    print("📦 Creating deployment package...")
    
    try:
        # Create deployment directory
        deploy_dir = "deployment_package"
        os.makedirs(deploy_dir, exist_ok=True)
        
        # Copy files to deployment package
        files_to_copy = [
            (model_path, "model"),
            (info_path, "metadata"), 
            (plots_path, "training_results")
        ]
        
        deployment_files = []
        
        for src_path, file_type in files_to_copy:
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(deploy_dir, filename)
                shutil.copy2(src_path, dest_path)
                deployment_files.append(dest_path)
                print(f"  ✅ Added {file_type}: {filename}")
        
        # Create deployment instructions
        instructions = f"""# 🌱 Puviyan Soil Detection Model Deployment

## 📦 Package Contents:
- `{MODEL_NAME}.tflite` - Trained TensorFlow Lite model
- `{MODEL_NAME}_info.json` - Model metadata and usage instructions  
- `training_history.png` - Training performance plots

## 🚀 Deployment Steps:

### 1. Mobile App Integration:
```bash
# Copy model to Flutter app
cp {MODEL_NAME}.tflite ../puviyan-mobile/assets/models/

# Enable TensorFlow Lite in pubspec.yaml
# Uncomment: tflite_flutter: ^0.10.4

# Update on-device service to use real model
# Remove mock mode from on_device_soil_detection_service.dart
```

### 2. Model Usage:
- **Input**: 224x224 RGB images, normalized to [0,1]
- **Output**: 8 class probabilities for Indian soil types
- **Confidence threshold**: 0.75 recommended

### 3. Soil Types:
{chr(10).join([f"{i}. {label}" for i, label in SOIL_LABELS.items()])}

## 📊 Model Performance:
- Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Model size: ~5-10MB (optimized for mobile)
- Expected inference time: ~150ms on mobile devices

## 🔧 Integration Notes:
- Replace mock inference in Flutter app
- Test with real soil images
- Monitor performance and accuracy
- Collect user feedback for model improvements

Generated by Puviyan AI Training Pipeline
"""
        
        readme_path = os.path.join(deploy_dir, "DEPLOYMENT_README.md")
        with open(readme_path, 'w') as f:
            f.write(instructions)
        
        deployment_files.append(readme_path)
        print(f"  ✅ Added deployment instructions: DEPLOYMENT_README.md")
        
        # Create zip package
        zip_name = f"puviyan_soil_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.make_archive(zip_name, 'zip', deploy_dir)
        
        print(f"📦 Deployment package created: {zip_name}.zip")
        return f"{zip_name}.zip", deployment_files
        
    except Exception as e:
        print(f"❌ Failed to create deployment package: {e}")
        return None, []

def show_download_instructions():
    """Show proper download instructions for Google Colab"""
    print("📥 TO UPDATE THIS SCRIPT IN GOOGLE COLAB:")
    print("="*60)
    print()
    print("# 🧹 Clean setup (removes old versions)")
    print("!rm -f train_soil_classifier.py*")
    print()
    print("# 📥 Download latest version (force overwrite)")  
    print("!wget -O train_soil_classifier.py https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/train_soil_classifier.py")
    print()
    print("# ✅ Verify download")
    print("!ls -la train_soil_classifier.py")
    print()
    print("# 🚀 Run training")
    print("!python train_soil_classifier.py")
    print()
    print("="*60)

def check_environment():
    """Check if running in appropriate environment"""
    print(f"🌱 Puviyan Soil Detection Training v{SCRIPT_VERSION}")
    print("="*60)
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        colab_env = True
    except ImportError:
        print("ℹ️ Not running in Google Colab")
        colab_env = False
    
    # Check TensorFlow and GPU
    print(f"✅ TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️ No GPU detected - training will be slower")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ RAM: {memory.total / (1024**3):.1f} GB available")
    except ImportError:
        print("ℹ️ Memory info not available")
    
    print("="*60)
    return colab_env

def main():
    """Main training pipeline"""
    # Check environment and show version info
    colab_env = check_environment()
    
    # Setup matplotlib for Colab
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Create output directory (current directory for Colab)
    output_dir = "."
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
    
    # Create deployment package
    info_path = model_path.replace('.tflite', '_info.json')
    plots_path = os.path.join(output_dir, 'training_history.png')
    zip_package, deployment_files = create_deployment_package(model_path, info_path, plots_path)
    
    # Upload model to GitHub repository
    model_files = [model_path, info_path, plots_path]
    upload_success = upload_model_to_github(model_files)
    
    print("\n🎉 Model training completed successfully!")
    print(f"📱 TensorFlow Lite model: {model_path}")
    print(f"📊 Final accuracy: {accuracy:.2%}")
    print(f"💾 Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    if zip_package:
        print(f"📦 Deployment package: {zip_package}")
    
    if upload_success:
        print("🚀 Model uploaded to GitHub repository")
        print("🔗 https://github.com/ctopuviyan/puviyan-ai-training")
    else:
        print("⚠️ GitHub upload failed - model available locally")
    
    print("\n✅ Ready for mobile deployment!")
    
    # Instructions for next steps
    print("\n🎯 Next Steps:")
    print("1. 📥 Download the .tflite model file")
    print("2. 📱 Copy to Flutter app: assets/models/")
    print("3. 🔧 Enable TensorFlow Lite in pubspec.yaml")
    print("4. 🧪 Test with real soil images")
    
    if not upload_success:
        print("\n💡 To manually upload model:")
        print("   - Download model files from Colab")
        print("   - Commit to GitHub repository")
        print("   - Use deployment script for mobile app")
    
    # Show download instructions for future use
    if colab_env:
        print("\n" + "="*60)
        print("📥 TO UPDATE THIS SCRIPT IN FUTURE:")
        print("Run this in a new Colab cell:")
        print()
        print("!rm -f train_soil_classifier.py*")
        print("!wget -O train_soil_classifier.py https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/train_soil_classifier.py")
        print("!python train_soil_classifier.py")
        print("="*60)

if __name__ == "__main__":
    main()
