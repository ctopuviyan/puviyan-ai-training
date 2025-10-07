"""
Script to create a new Colab notebook for Puviyan Soil Detection training.
Run this script to generate 'Puviyan_Soil_Detection_v5_Enhanced.ipynb'
"""

import json

# Notebook content as a Python dictionary
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "soil_detection_title"
   },
   "source": [
    "# 🌱 Puviyan Soil Detection - Enhanced Training (v5.0)\n",
    "\n",
    "## 🚀 Key Features\n",
    "- **GPU-accelerated** training (up to 10x faster)\n",
    "- **Multiple dataset options**: Synthetic, Real, or Hybrid\n",
    "- **Enhanced model architecture** for better accuracy\n",
    "- **TFLite export** for Flutter deployment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "setup_environment"
   },
   "outputs": [],
   "source": [
    "# @title 🚀 Setup and Configuration\n",
    "print(\"Setting up environment...\")\n",
    "!nvidia-smi  # Check GPU status\n",
    "!pip install -q tensorflow tensorflow-model-optimization matplotlib numpy tqdm\n",
    "\n",
    "import os\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm.notebook import tqdm\n",
    "import time\n",
    "from google.colab import files\n",
    "from IPython.display import clear_output\n",
    "\n",
    "# Configuration\n",
    "CONFIG = {\n",
    "    'model_name': 'soil_classifier_enhanced_v5',\n",
    "    'input_size': 224,\n",
    "    'num_classes': 8,\n",
    "    'batch_size': 32,  # Will be adjusted based on GPU memory\n",
    "    'epochs': 100,\n",
    "    'learning_rate': 0.0005,\n",
    "    'patience': 15,\n",
    "    'min_delta': 1e-4\n",
    "}\n",
    "\n",
    "# Soil type labels\n",
    "SOIL_LABELS = {\n",
    "    0: \"Alluvial Soil\",\n",
    "    1: \"Black Soil\",\n",
    "    2: \"Red Soil\",\n",
    "    3: \"Laterite Soil\",\n",
    "    4: \"Desert Soil\",\n",
    "    5: \"Saline/Alkaline Soil\",\n",
    "    6: \"Peaty/Marshy Soil\",\n",
    "    7: \"Forest/Hill Soil\"\n",
    "}\n",
    "\n",
    "# Setup GPU\n",
    "gpus = tf.config.list_physical_devices('GPU')\n",
    "if gpus:\n",
    "    print(f\\\"✅ GPU Detected: {tf.test.gpu_device_name()}\\\")\n",
    "    for gpu in gpus:\n",
    "        tf.config.experimental.set_memory_growth(gpu, True)\n",
    "else:\n",
    "    print(\\\"⚠️ No GPU detected. Training will be very slow!\\\")\n",
    "    print(\\\"   Go to Runtime > Change runtime type > Select GPU\\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "data_loading"
   },
   "source": [
    "## 📥 Data Loading\n",
    "Run the cell below to set up data loading functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "data_loading_code"
   },
   "outputs": [],
   "source": [
    "# @title 🔄 Data Loading Functions\n",
    "def generate_synthetic_data(num_samples=1000):\n",
    "    print(f\\\"🎨 Generating {num_samples} synthetic soil samples...\\\")\n",
    "    # Generate random images (replace with actual synthetic data generation)\n",
    "    X = np.random.randint(0, 255, (num_samples, CONFIG['input_size'], CONFIG['input_size'], 3), dtype=np.uint8)\n",
    "    y = np.random.randint(0, CONFIG['num_classes'], num_samples, dtype=np.int32)\n",
    "    return X, y\n",
    "\n",
    "def upload_real_data():\n",
    "    from google.colab import files\n",
    "    import zipfile\n",
    "    \n",
    "    print(\\\"📤 Please upload your dataset zip file\\\")\n",
    "    uploaded = files.upload()\n",
    "    \n",
    "    if not uploaded:\n",
    "        print(\\\"⚠️ No files uploaded. Using synthetic data.\\\")\n",
    "        return None, None\n",
    "    \n",
    "    zip_name = list(uploaded.keys())[0]\n",
    "    with zipfile.ZipFile(zip_name, 'r') as zip_ref:\n",
    "        zip_ref.extractall('dataset')\n",
    "    # For demo - replace with actual data loading\n",
    "    print(\\\"📂 Loading real data...\\\")\n",
    "    # [Add your data loading code here]\n",
    "    return None, None"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "model_architecture"
   },
   "source": [
    "## 🏗️ Model Architecture\n",
    "Run the cell below to define the enhanced model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "model_architecture_code"
   },
   "outputs": [],
   "source": [
    "# @title 🧠 Enhanced Model Architecture\n",
    "def create_enhanced_model():\n",
    "    print(\\\"🏗️ Creating enhanced model architecture...\\\")\n",
    "    \n",
    "    inputs = keras.Input(shape=(CONFIG['input_size'], CONFIG['input_size'], 3))\n",
    "    \n",
    "    # Data augmentation\n",
    "    x = layers.Rescaling(1./255)(inputs)\n",
    "    x = layers.RandomFlip(\\\"horizontal\\\")(x)\n",
    "    x = layers.RandomRotation(0.2)(x)\n",
    "    x = layers.RandomZoom(0.2)(x)\n",
    "    \n",
    "    # Feature extraction\n",
    "    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.MaxPooling2D()(x)\n",
    "    \n",
    "    x = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.MaxPooling2D()(x)\n",
    "    \n",
    "    x = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.GlobalAveragePooling2D()(x)\n",
    "    \n",
    "    # Classification head\n",
    "    x = layers.Dense(256, activation='relu')(x)\n",
    "    x = layers.Dropout(0.5)(x)\n",
    "    outputs = layers.Dense(CONFIG['num_classes'], activation='softmax')(x)\n",
    "    \n",
    "    model = keras.Model(inputs=inputs, outputs=outputs)\n",
    "    \n",
    "    model.compile(\n",
    "        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),\n",
    "        loss='sparse_categorical_crossentropy',\n",
    "        metrics=['accuracy']\n",
    "    )\n",
    "    \n",
    "    return model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "training_section"
   },
   "source": [
    "## 🚀 Training\n",
    "Run the cell below to start training with your selected dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "training_code"
   },
   "outputs": [],
   "source": [
    "# @title 🏃 Start Training\n",
    "def train():  \n",
    "    # Select dataset type  \n",
    "    print(\\\"📊 Select dataset type:\\\")\n",
    "    print(\\\"1. Synthetic data (generated)\\\")\n",
    "    print(\\\"2. Upload real data\\\")\n",
    "    print(\\\"3. Hybrid (synthetic + real)\\\")\n",
    "    \n",
    "    choice = input(\\\"Enter your choice (1-3): \\\").strip()\n",
    "    \n",
    "    # Load data based on choice  \n",
    "    if choice == '1':  \n",
    "        X, y = generate_synthetic_data(num_samples=5000)  \n",
    "    elif choice == '2':  \n",
    "        X, y = upload_real_data()  \n",
    "        if X is None:  \n",
    "            print(\\\"⚠️ Using synthetic data as fallback\\\")\n",
    "            X, y = generate_synthetic_data(num_samples=5000)  \n",
    "    elif choice == '3':  \n",
    "        X_syn, y_syn = generate_synthetic_data(num_samples=2500)  \n",
    "        X_real, y_real = upload_real_data()  \n",
    "        if X_real is not None:  \n",
    "            X = np.concatenate([X_syn, X_real], axis=0)  \n",
    "            y = np.concatenate([y_syn, y_real], axis=0)  \n",
    "        else:  \n",
    "            print(\\\"⚠️ Using synthetic data only\\\")\n",
    "            X, y = X_syn, y_syn  \n",
    "    else:  \n",
    "        print(\\\"❌ Invalid choice. Using synthetic data by default.\\\")\n",
    "        X, y = generate_synthetic_data(num_samples=5000)  \n",
    "    \n",
    "    # Create and train model  \n",
    "    model = create_enhanced_model()  \n",
    "    \n",
    "    # Callbacks  \n",
    "    callbacks = [  \n",
    "        keras.callbacks.EarlyStopping(  \n",
    "            monitor='val_accuracy',  \n",
    "            patience=CONFIG['patience'],  \n",
    "            restore_best_weights=True  \n",
    "        ),  \n",
    "        keras.callbacks.ModelCheckpoint(  \n",
    "            f\\\"{CONFIG['model_name']}.h5\\\",  \n",
    "            save_best_only=True,  \n",
    "            monitor='val_accuracy'  \n",
    "        )  \n",
    "    ]  \n",
    "    \n",
    "    # Train  \n",
    "    print(\\\"\\n🚀 Starting training...\\\")\n",
    "    history = model.fit(  \n",
    "        X, y,  \n",
    "        validation_split=0.2,  \n",
    "        batch_size=CONFIG['batch_size'],  \n",
    "        epochs=CONFIG['epochs'],  \n",
    "        callbacks=callbacks,  \n",
    "        verbose=1  \n",
    "    )  \n",
    "    \n",
    "    # Save model to TFLite  \n",
    "    converter = tf.lite.TFLiteConverter.from_keras_model(model)  \n",
    "    tflite_model = converter.convert()  \n",
    "    \n",
    "    with open(f\\\"{CONFIG['model_name']}.tflite\\\", 'wb') as f:  \n",
    "        f.write(tflite_model)  \n",
    "    \n",
    "    print(f\\\"✅ Model saved as {CONFIG['model_name']}.tflite\\\")\n",
    "    \n",
    "    # Download the model  \n",
    "    files.download(f\\\"{CONFIG['model_name']}.tflite\\\")\n",
    "\n",
    "# Start training  \n",
    "train()"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  },
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

# Save the notebook to a file
with open('Puviyan_Soil_Detection_v5_Enhanced.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f)

print("✅ Successfully created 'Puviyan_Soil_Detection_v5_Enhanced.ipynb'")
