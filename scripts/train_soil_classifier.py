#!/usr/bin/env python3
"""
Puviyan Soil Detection Training Script
========================================

Advanced soil classification model training for Indian soil types using TensorFlow.
Supports both real soil image datasets and synthetic data generation.

NEW: Real Dataset Support!
- Upload ZIP files with real soil images
- Automatic folder-to-soil-type mapping
- Enhanced accuracy with real-world data
- Fallback to synthetic data generation

Features:
- 8 Indian soil types classification
- Real dataset upload and processing
- Synthetic data generation with realistic soil characteristics
- TensorFlow Lite model export for mobile apps
- Comprehensive training metrics and visualization
- Google Colab optimized with automatic file downloads

Indian Soil Types Supported:
1. Alluvial Soil - Fertile river deposits, ideal for crops
2. Black Soil - Clay-rich, excellent for cotton cultivation  
3. Red Soil - Iron-rich, good drainage, suitable for groundnuts
4. Laterite Soil - tropical weathered soil, good for cashews
5. Desert Soil - Arid conditions, limited agriculture potential
6. Saline/Alkaline Soil - High salt content, needs soil treatment
7. Peaty/Marshy Soil - Organic-rich, waterlogged conditions
8. Forest/Hill Soil - Rich humus content, ideal for plantation crops

Usage in Google Colab:
1. Clean up any existing files (PREVENTS .py.1, .py.2 files):
   import os
   for f in os.listdir('.'):
       if f.startswith('train_soil_classifier.py'):
           os.remove(f)

2. Download this script (FORCE OVERWRITE):
   !wget -O train_soil_classifier.py https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/train_soil_classifier.py

3. Verify download (should show v3.3.0):
   !head -20 train_soil_classifier.py | grep "3.3.0"

4. Run training (will prompt for dataset choice):
   !python train_soil_classifier.py

5. Choose: Real dataset upload OR synthetic generation

6. Download generated files (model, metadata, plots)

Author: Puviyan AI Team
Version: 3.3.0 (Runtime Processing Mode Selection + Full GPU Utilization)
License: MIT
Repository: https://github.com/ctopuviyan/puviyan-ai-training
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
from sklearn.model_selection import train_test_split
import random

# Configuration
MODEL_NAME = "soil_classifier_lite"
INPUT_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 32  # Optimized for GPU (increased from 16)
EPOCHS = 30      # Reduced for faster training
LEARNING_RATE = 0.001
SCRIPT_VERSION = "3.3.0"  # Added runtime processing mode selection (batch vs full GPU)
MAX_IMAGES_PER_CLASS = 200  # Limit images per class to prevent memory issues

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

class SoilDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator for soil images"""
    
    def __init__(self, image_paths, labels, batch_size=16, input_size=224, shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_size = len(batch_indices)
        X = np.zeros((batch_size, self.input_size, self.input_size, 3), dtype=np.float32)
        y = np.zeros((batch_size, NUM_CLASSES), dtype=np.float32)
        
        # Load and process images
        for i, idx in enumerate(batch_indices):
            try:
                # Load image
                from PIL import Image
                img = Image.open(self.image_paths[idx]).convert('RGB')
                
                # Resize image
                img = img.resize((self.input_size, self.input_size))
                
                # Convert to array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Apply augmentation if enabled
                if self.augment:
                    img_array = self._augment_image(img_array)
                
                X[i] = img_array
                y[i] = tf.keras.utils.to_categorical(self.labels[idx], NUM_CLASSES)
                
            except Exception as e:
                print(f"⚠️ Error loading image {self.image_paths[idx]}: {e}")
                # Fill with zeros if image fails to load
                X[i] = np.zeros((self.input_size, self.input_size, 3))
                y[i] = tf.keras.utils.to_categorical(0, NUM_CLASSES)
        
        return X, y
    
    def _augment_image(self, img_array):
        """Apply simple data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img_array = np.fliplr(img_array)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            img_array = np.clip(img_array * brightness_factor, 0, 1)
        
        return img_array
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

def setup_dataset_choice():
    """Setup dataset choice for Colab users"""
    try:
        import google.colab
        
        print("📊 Dataset Options:")
        print("1. Upload real soil images (ZIP file) - Automatic")
        print("2. Upload real soil images (ZIP file) - Manual method")
        print("3. Use synthetic dataset generation")
        print()
        
        choice = input("Choose dataset type (1 for auto, 2 for manual, 3 for synthetic): ").strip()
        
        if choice == "1":
            return setup_real_dataset_upload()
        elif choice == "2":
            return setup_manual_dataset_upload()
        else:
            print("🎨 Using synthetic dataset generation...")
            return False
            
    except ImportError:
        print("ℹ️ Not in Colab - checking for local dataset...")
        return check_local_dataset()

def setup_processing_mode():
    """Setup processing mode for GPU optimization"""
    print("\n⚡ Processing Mode Options:")
    print("1. 🔋 Memory-Efficient Mode (Batch Processing)")
    print("   - Uses data generators")
    print("   - Lower GPU utilization but handles large datasets")
    print("   - Recommended for datasets > 1000 images")
    print("   - Memory usage: ~100MB")
    print()
    print("2. 🚀 GPU-Optimized Mode (Full Loading)")
    print("   - Loads all images into GPU memory")
    print("   - Maximum GPU utilization and speed")
    print("   - Recommended for datasets < 1000 images")
    print("   - Memory usage: ~2-4GB")
    print()
    print("3. 🎯 Auto Mode (Recommended)")
    print("   - Automatically chooses based on dataset size")
    print("   - < 1000 images: GPU-Optimized")
    print("   - > 1000 images: Memory-Efficient")
    print()
    
    mode_choice = input("Choose processing mode (1 for memory-efficient, 2 for GPU-optimized, 3 for auto): ").strip()
    
    if mode_choice == "1":
        print("🔋 Selected: Memory-Efficient Mode (Batch Processing)")
        return "batch"
    elif mode_choice == "2":
        print("🚀 Selected: GPU-Optimized Mode (Full Loading)")
        return "full"
    else:
        print("🎯 Selected: Auto Mode (Will decide based on dataset size)")
        return "auto"

def setup_manual_dataset_upload():
    """Handle manual dataset upload in Colab (bypass automatic upload)"""
    print("📁 MANUAL DATASET UPLOAD METHOD")
    print("=" * 50)
    
    try:
        import google.colab
        print("✅ Google Colab environment detected")
        
        print("\n📋 STEP-BY-STEP MANUAL UPLOAD:")
        print("1. 📁 Click the folder icon on the LEFT SIDEBAR")
        print("2. 📤 Click the upload button (folder with up arrow)")
        print("3. 🗂️ Select your soil_dataset.zip file from your computer")
        print("4. ⏳ Wait for the upload progress bar to complete")
        print("5. ✅ Your file should appear in the file list")
        print()
        
        print("📋 Expected ZIP structure:")
        for i, soil_type in SOIL_LABELS.items():
            folder_name = soil_type.replace('/', '_').replace(' ', '_')
            print(f"   📁 {folder_name}/ (Class {i})")
        
        print("\n🛑 AFTER UPLOADING YOUR ZIP FILE:")
        print("Press Enter to continue and the script will look for your uploaded file...")
        input("Press Enter after you've uploaded your ZIP file: ")
        
        # Check if file was uploaded manually
        import os
        zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
        
        if zip_files:
            print(f"✅ Found {len(zip_files)} ZIP file(s): {zip_files}")
            for zip_file in zip_files:
                if 'soil' in zip_file.lower() or 'dataset' in zip_file.lower():
                    print(f"📦 Processing: {zip_file}")
                    return extract_and_process_dataset(zip_file)
            
            # If no soil-related ZIP found, use the first one
            print(f"📦 Processing: {zip_files[0]}")
            return extract_and_process_dataset(zip_files[0])
        else:
            print("❌ No ZIP files found in the current directory")
            print("💡 Make sure you uploaded the file to the root directory (not in a subfolder)")
            print("🔄 Try uploading again or use synthetic data")
            return False
            
    except ImportError:
        print("❌ Not running in Google Colab")
        return False
    except Exception as e:
        print(f"❌ Manual upload process failed: {e}")
        return False

def setup_real_dataset_upload():
    """Handle real dataset upload in Colab"""
    print("📤 REAL DATASET UPLOAD FROM LOCAL MACHINE")
    print("=" * 50)
    
    try:
        # Check if we're in Colab
        import google.colab
        print("✅ Google Colab environment detected")
        
        # Import files module
        from google.colab import files
        import zipfile
        
        print("\n📋 DATASET PREPARATION GUIDE:")
        print("1. Create folders for each soil type on your computer:")
        for i, soil_type in SOIL_LABELS.items():
            folder_name = soil_type.replace('/', '_').replace(' ', '_')
            print(f"   📁 {folder_name}/ (Class {i})")
        
        print("\n2. Put soil images in respective folders:")
        print("   📁 Alluvial_Soil/")
        print("      ├── 📸 soil_image_1.jpg")
        print("      ├── 📸 soil_image_2.jpg")
        print("      └── 📸 ...")
        print("   📁 Black_Soil/")
        print("      ├── 📸 black_soil_1.jpg")
        print("      └── 📸 ...")
        
        print("\n3. Create a ZIP file containing all folders")
        print("4. Upload the ZIP file using the button below")
        
        print("\n" + "=" * 50)
        print("🔄 CLICK THE UPLOAD BUTTON BELOW:")
        print("=" * 50)
        
        # Force flush output to ensure messages appear
        import sys
        sys.stdout.flush()
        
        # Trigger file upload with better error handling
        print("📤 Waiting for file upload...")
        
        # Try multiple approaches for file upload
        uploaded = None
        try:
            # Method 1: Standard upload
            uploaded = files.upload()
        except Exception as e1:
            print(f"⚠️ Standard upload failed: {e1}")
            print("🔄 Trying alternative upload method...")
            
            try:
                # Method 2: Force display and upload
                from IPython.display import display, HTML
                display(HTML('<p>📤 Please use the file upload button below:</p>'))
                uploaded = files.upload()
            except Exception as e2:
                print(f"⚠️ Alternative upload failed: {e2}")
                print("🔄 Trying manual approach...")
                
                try:
                    # Method 3: Manual widget approach
                    import ipywidgets as widgets
                    from IPython.display import display
                    
                    print("📤 Manual file upload - click 'Choose Files' button:")
                    uploader = widgets.FileUpload(
                        accept='.zip',
                        multiple=False,
                        description='Upload ZIP'
                    )
                    display(uploader)
                    
                    # Wait for upload (this is a simplified approach)
                    print("⏳ After selecting your file, run the next cell to continue...")
                    print("💡 Or try restarting runtime and running again")
                    print()
                    print("🛑 SCRIPT PAUSED - Choose one of these options:")
                    print("1. 🔄 Restart runtime and try again")
                    print("2. 📁 Use manual upload method (see instructions above)")
                    print("3. 🎨 Continue with synthetic data (type 'synthetic' and press Enter)")
                    
                    # Wait for user input
                    user_choice = input("Enter your choice (restart/manual/synthetic): ").strip().lower()
                    
                    if user_choice in ['synthetic', 's', '2']:
                        print("🎨 Continuing with synthetic dataset...")
                        return False
                    elif user_choice in ['manual', 'm']:
                        print("📋 Please follow the manual upload instructions above")
                        print("🔄 After uploading, restart the script and it should detect your dataset")
                        raise SystemExit("Manual upload chosen - please restart script after uploading")
                    else:
                        print("🔄 Please restart runtime and try again")
                        raise SystemExit("Restart requested - please restart runtime")
                    
                except Exception as e3:
                    print(f"⚠️ Manual upload also failed: {e3}")
                    print("🔧 TROUBLESHOOTING STEPS:")
                    print("1. Restart runtime: Runtime > Restart runtime")
                    print("2. Clear outputs: Edit > Clear all outputs") 
                    print("3. Run cells again from the beginning")
                    print("4. Try a different browser (Chrome recommended)")
                    print("5. Check if pop-ups are blocked")
                    print()
                    print("📋 ALTERNATIVE: Use manual upload method")
                    manual_upload_instructions()
                    return False
        
        if not uploaded:
            print("❌ No files were uploaded.")
            print("💡 Make sure to click 'Choose Files' and select your ZIP file")
            return False
        
        print(f"✅ {len(uploaded)} file(s) uploaded successfully!")
        
        # Process uploaded files
        for filename, content in uploaded.items():
            print(f"📄 Processing: {filename} ({len(content)} bytes)")
            
            if filename.endswith('.zip'):
                print(f"📦 Extracting ZIP file: {filename}")
                return extract_and_process_dataset(filename)
            else:
                print(f"⚠️ {filename} is not a ZIP file")
        
        print("❌ No ZIP files found in upload. Using synthetic dataset.")
        return False
        
    except ImportError:
        print("❌ Not running in Google Colab")
        print("💡 This function requires Google Colab environment")
        print("💡 For local development, place dataset in 'soil_dataset' folder")
        return False
        
    except Exception as e:
        print(f"❌ Upload failed with error: {e}")
        print("💡 Try refreshing the page and running again")
        print("💡 Make sure your ZIP file is not too large (< 25MB recommended)")
        return False

def manual_upload_instructions():
    """Provide manual upload instructions as fallback"""
    print("📋 MANUAL UPLOAD ALTERNATIVE")
    print("=" * 40)
    print("If automatic upload fails, try this approach:")
    print()
    print("1. 📁 In Colab, click the folder icon on the left sidebar")
    print("2. 📤 Click the upload button (folder with up arrow)")
    print("3. 🗂️ Select your soil_dataset.zip file")
    print("4. ⏳ Wait for upload to complete")
    print("5. ✅ Your file will appear in the file list")
    print()
    print("Then run this code to process your uploaded dataset:")
    print("```python")
    print("# Process manually uploaded dataset")
    print("import zipfile")
    print("import os")
    print()
    print("# Extract the ZIP file")
    print("with zipfile.ZipFile('soil_dataset.zip', 'r') as zip_ref:")
    print("    zip_ref.extractall('real_soil_dataset')")
    print()
    print("# List extracted contents")
    print("for root, dirs, files in os.walk('real_soil_dataset'):")
    print("    level = root.replace('real_soil_dataset', '').count(os.sep)")
    print("    indent = ' ' * 2 * level")
    print("    print(f'{indent}{os.path.basename(root)}/')")
    print("    subindent = ' ' * 2 * (level + 1)")
    print("    for file in files[:3]:  # Show first 3 files")
    print("        print(f'{subindent}{file}')")
    print("```")

def test_colab_upload():
    """Simple test function to verify Colab file upload works"""
    print("🧪 TESTING COLAB FILE UPLOAD")
    print("=" * 40)
    
    try:
        import google.colab
        from google.colab import files
        
        print("✅ Google Colab detected")
        print("✅ Files module imported")
        print("\n📤 Test upload - select any small file:")
        
        uploaded = files.upload()
        
        if uploaded:
            for filename, content in uploaded.items():
                print(f"✅ Successfully uploaded: {filename} ({len(content)} bytes)")
            return True
        else:
            print("❌ No files uploaded")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_local_dataset():
    """Check for local dataset directory"""
    dataset_paths = ['soil_dataset', 'real_soil_dataset', 'dataset', 'data']
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Found local dataset at: {path}")
            return process_local_dataset(path)
    
    print("ℹ️ No local dataset found. Using synthetic generation.")
    return False

def extract_and_process_dataset(zip_filename):
    """Extract and process uploaded dataset"""
    import zipfile
    from PIL import Image
    
    dataset_dir = "real_soil_dataset"
    
    # Clean existing dataset
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    return process_local_dataset(dataset_dir)

def process_local_dataset(dataset_dir):
    """Process local dataset directory - Memory efficient version"""
    print(f"📊 Processing dataset in {dataset_dir}...")
    
    # Find image files and map to soil types
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []
    labels = []
    
    print(f"🔍 Looking for soil type folders...")
    
    for root, dirs, files in os.walk(dataset_dir):
        folder_name = os.path.basename(root).lower()
        
        # Skip the root dataset directory and __MACOSX folders
        if root == dataset_dir or '__MACOSX' in root:
            if root == dataset_dir:
                print(f"📁 Root directory contains: {dirs}")
            continue
        
        print(f"🔍 Checking folder: '{folder_name}'")
        
        # Map folder name to soil type
        soil_class = None
        matched_keyword = None
        for keyword, class_id in FOLDER_MAPPINGS.items():
            if keyword in folder_name:
                soil_class = class_id
                matched_keyword = keyword
                break
        
        if soil_class is None:
            print(f"   ❌ No mapping found for folder '{folder_name}'")
            print(f"   💡 Available keywords: {list(FOLDER_MAPPINGS.keys())}")
            continue
        
        print(f"   ✅ Mapped '{folder_name}' → {SOIL_LABELS[soil_class]} (keyword: '{matched_keyword}')")
        
        # Get image files
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
        print(f"   📸 Found {len(image_files)} image files")
        
        if not image_files:
            print(f"   ⚠️ No image files found in {folder_name}")
            continue
        
        # Limit images per class to prevent memory issues
        if len(image_files) > MAX_IMAGES_PER_CLASS:
            print(f"   📊 Limiting to {MAX_IMAGES_PER_CLASS} images (from {len(image_files)})")
            image_files = random.sample(image_files, MAX_IMAGES_PER_CLASS)
        
        # Add image paths and labels (don't load images yet)
        class_image_count = 0
        for filename in image_files:
            img_path = os.path.join(root, filename)
            
            # Quick validation - check if file exists and is readable
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    # Just check if we can open it, don't load into memory
                    pass
                
                image_paths.append(img_path)
                labels.append(soil_class)
                class_image_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Skipping {filename}: {e}")
                continue
        
        print(f"   ✅ Added {class_image_count} valid images from {SOIL_LABELS[soil_class]}")
    
    if not image_paths:
        print("\n❌ No valid images found!")
        print("💡 Expected folder structure:")
        for keyword, class_id in FOLDER_MAPPINGS.items():
            print(f"   📁 {keyword} → {SOIL_LABELS[class_id]}")
        return False
    
    # Store paths globally instead of loaded images
    global real_dataset_paths, real_dataset_labels
    real_dataset_paths = image_paths
    real_dataset_labels = labels
    
    print(f"✅ Dataset prepared: {len(image_paths)} images, {len(np.unique(labels))} classes")
    
    # Show class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("📊 Class distribution:")
    for class_id, count in zip(unique, counts):
        print(f"   {SOIL_LABELS[class_id]}: {count} images")
    
    return True

def load_full_dataset_to_memory(image_paths, labels):
    """Load full dataset into memory for GPU optimization"""
    print("🚀 Loading full dataset into GPU memory for maximum performance...")
    
    from PIL import Image
    
    total_images = len(image_paths)
    print(f"📊 Loading {total_images} images into memory...")
    
    # Pre-allocate arrays
    X = np.zeros((total_images, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    y = np.zeros(total_images, dtype=np.int32)
    
    # Load images with progress tracking
    loaded_count = 0
    failed_count = 0
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((INPUT_SIZE, INPUT_SIZE))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            X[i] = img_array
            y[i] = label
            loaded_count += 1
            
            # Show progress every 100 images
            if (i + 1) % 100 == 0 or (i + 1) == total_images:
                progress = (i + 1) / total_images * 100
                print(f"   📸 Progress: {i + 1}/{total_images} ({progress:.1f}%) - {loaded_count} loaded, {failed_count} failed")
                
        except Exception as e:
            print(f"   ⚠️ Failed to load {img_path}: {e}")
            # Fill with zeros for failed images
            X[i] = np.zeros((INPUT_SIZE, INPUT_SIZE, 3))
            y[i] = 0
            failed_count += 1
    
    print(f"✅ Dataset loaded: {loaded_count} images successfully, {failed_count} failed")
    print(f"📊 Memory usage: ~{X.nbytes / (1024**2):.1f} MB")
    
    return X, y

# Global variables for real dataset (now using paths instead of loaded images)
real_dataset_paths = None
real_dataset_labels = None

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

def create_model(use_sparse_labels=False):
    """Create the soil classification model optimized for GPU"""
    print("🏗️ Creating soil classification model...")
    
    # Enable mixed precision for GPU optimization (if available)
    try:
        if tf.config.list_physical_devices('GPU'):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled for faster GPU training")
    except:
        print("ℹ️ Mixed precision not available, using float32")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        
        # Feature extraction layers (GPU-optimized with BatchNorm)
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
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')  # Keep output as float32
    ])
    
    # Compile model with appropriate loss function
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Choose loss function based on label format
    loss_function = 'sparse_categorical_crossentropy' if use_sparse_labels else 'categorical_crossentropy'
    print(f"📊 Using loss function: {loss_function}")
    
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )
    
    print("✅ Model created with GPU optimizations")
    print(f"📊 Total parameters: {model.count_params():,}")
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

def train_model_with_generators(model, train_generator, val_generator):
    """Train the soil classification model using data generators"""
    print("🚀 Starting model training with data generators...")
    
    try:
        print(f"📊 Training batches: {len(train_generator)}")
        print(f"📊 Validation batches: {len(val_generator)}")
        print(f"📊 Batch size: {BATCH_SIZE}")
        print(f"📊 Epochs: {EPOCHS}")
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("🎯 Starting training process...")
        
        # Check GPU utilization before training
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"⚡ Training with GPU acceleration on {len(gpus)} device(s)")
            print(f"⏱️ Estimated time: {EPOCHS} epochs × ~2 minutes = ~{EPOCHS*2} minutes")
        else:
            print("🐌 Training with CPU (no GPU detected)")
            print(f"⏱️ Estimated time: {EPOCHS} epochs × ~15 minutes = ~{EPOCHS*15} minutes")
        
        # Train model with generators (GPU optimized)
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed successfully!")
        return history
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C)")
        print("💡 You can:")
        print("1. Restart the script and choose synthetic data for faster training")
        print("2. Reduce EPOCHS or MAX_IMAGES_PER_CLASS in the script")
        raise
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("🔍 Debugging info:")
        print(f"   - Training batches: {len(train_generator) if 'train_generator' in locals() else 'Not created'}")
        print(f"   - Validation batches: {len(val_generator) if 'val_generator' in locals() else 'Not created'}")
        print("💡 Possible solutions:")
        print(f"1. Reduce MAX_IMAGES_PER_CLASS (currently {MAX_IMAGES_PER_CLASS})")
        print(f"2. Reduce BATCH_SIZE (currently {BATCH_SIZE})")
        print("3. Try with synthetic data first to test the pipeline")
        raise

def convert_to_tflite(model, output_path):
    """Convert Keras model to TensorFlow Lite"""
    print("📱 Converting to TensorFlow Lite...")
    
    # Convert to TensorFlow Lite with TF Select ops for compatibility
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable TF Select ops to handle BatchNormalization and other ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow ops (for BatchNorm, etc.)
    ]
    
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

def test_tflite_model_with_generator(tflite_model, val_generator):
    """Test the TensorFlow Lite model using data generator"""
    print("🧪 Testing TensorFlow Lite model with generator...")
    
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct_predictions = 0
    total_predictions = 0
    
    print(f"📊 Testing on {len(val_generator)} batches...")
    
    # Test on validation generator
    for batch_idx in range(len(val_generator)):
        X_batch, y_batch = val_generator[batch_idx]
        
        for i in range(len(X_batch)):
            # Prepare input
            input_data = np.expand_dims(X_batch[i], axis=0).astype(np.float32)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get prediction
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            true_class = np.argmax(y_batch[i])
            
            if predicted_class == true_class:
                correct_predictions += 1
            total_predictions += 1
        
        # Show progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            current_accuracy = correct_predictions / total_predictions
            print(f"   Batch {batch_idx + 1}/{len(val_generator)}: {current_accuracy:.3f} accuracy")
    
    accuracy = correct_predictions / total_predictions
    print(f"✅ TensorFlow Lite model accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
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
    print("⚠️  IMPORTANT: Prevents incremental filenames (.py.1, .py.2)")
    print()
    print("# 🧹 Complete cleanup (prevents .py.1, .py.2 files)")
    print("import os")
    print("for f in os.listdir('.'):")
    print("    if f.startswith('train_soil_classifier.py'):")
    print("        os.remove(f)")
    print("        print(f'🗑️ Removed: {f}')")
    print()
    print("# 📥 Download latest version (FORCE OVERWRITE)")  
    print("!wget -O train_soil_classifier.py https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/train_soil_classifier.py")
    print()
    print("# ✅ Verify download (should show v3.3.0)")
    print("!ls -la train_soil_classifier.py")
    print("!head -20 train_soil_classifier.py | grep '3.3.0'")
    print()
    print("# 🚀 Run training (will show dataset choice)")
    print("!python train_soil_classifier.py")
    print()
    print("✅ This prevents the common .py.1, .py.2 filename issue!")
    print("="*60)

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
    
    # Check TensorFlow version
    print(f"✅ TensorFlow: {tf.__version__}")
    
    # Setup GPU optimization
    gpu_available = setup_gpu_optimization()
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    except ImportError:
        print("ℹ️ Memory info not available")
    
    # Performance recommendations
    if gpu_available:
        print("🚀 Performance mode: GPU-accelerated training")
        print(f"⚡ Expected training time: ~{EPOCHS * 2} minutes with GPU")
    else:
        print("🐌 Performance mode: CPU-only training")
        print(f"⏱️ Expected training time: ~{EPOCHS * 15} minutes with CPU")
    
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
    
    # Setup dataset choice
    use_real_data = setup_dataset_choice()
    
    # Setup processing mode if using real data
    processing_mode = "batch"  # default
    if use_real_data and real_dataset_paths is not None:
        processing_mode = setup_processing_mode()
    
    # Create dataset
    if use_real_data and real_dataset_paths is not None:
        print("🚀 Using real soil dataset for training...")
        
        # Decide processing mode based on dataset size and user choice
        dataset_size = len(real_dataset_paths)
        
        if processing_mode == "auto":
            if dataset_size < 1000:
                processing_mode = "full"
                print(f"🎯 Auto mode: Dataset size ({dataset_size}) < 1000, using GPU-Optimized mode")
            else:
                processing_mode = "batch"
                print(f"🎯 Auto mode: Dataset size ({dataset_size}) >= 1000, using Memory-Efficient mode")
        
        # Split dataset paths and labels
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            real_dataset_paths, real_dataset_labels,
            test_size=0.2,
            random_state=42,
            stratify=real_dataset_labels
        )
        
        print(f"📊 Dataset split:")
        print(f"   Training: {len(train_paths)} samples")
        print(f"   Validation: {len(val_paths)} samples")
        
        if processing_mode == "full":
            # GPU-Optimized Mode: Load all data into memory
            print("🚀 GPU-Optimized Mode: Loading full dataset into memory...")
            
            # Load training data
            X_train, y_train = load_full_dataset_to_memory(train_paths, train_labels)
            X_val, y_val = load_full_dataset_to_memory(val_paths, val_labels)
            
            train_data = (X_train, y_train)
            val_data = (X_val, y_val)
            use_generators = False
            
            print("✅ Full dataset loaded into GPU memory for maximum performance!")
            
        else:
            # Memory-Efficient Mode: Use data generators
            print("🔋 Memory-Efficient Mode: Using data generators...")
            
            train_generator = SoilDataGenerator(
                train_paths, train_labels,
                batch_size=BATCH_SIZE,
                input_size=INPUT_SIZE,
                shuffle=True,
                augment=True  # Enable augmentation for training
            )
            
            val_generator = SoilDataGenerator(
                val_paths, val_labels,
                batch_size=BATCH_SIZE,
                input_size=INPUT_SIZE,
                shuffle=False,
                augment=False  # No augmentation for validation
            )
            
            print(f"✅ Generators created:")
            print(f"   Training batches: {len(train_generator)}")
            print(f"   Validation batches: {len(val_generator)}")
            
            use_generators = True
        
    else:
        print("🎨 Using synthetic dataset generation...")
        train_data, val_data = create_synthetic_dataset()
        use_generators = False
    
    # Create and train model
    if use_generators:
        # Real dataset with generators uses one-hot encoded labels (categorical_crossentropy)
        model = create_model(use_sparse_labels=False)
        history = train_model_with_generators(model, train_generator, val_generator)
    else:
        # Full dataset or synthetic uses integer labels (sparse_categorical_crossentropy)
        model = create_model(use_sparse_labels=True)
        history = train_model(model, train_data, val_data)
    
    # Convert to TensorFlow Lite
    model_path = os.path.join(output_dir, f"{MODEL_NAME}.tflite")
    tflite_model = convert_to_tflite(model, model_path)
    
    # Test TensorFlow Lite model
    if use_generators:
        accuracy = test_tflite_model_with_generator(tflite_model, val_generator)
    else:
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
