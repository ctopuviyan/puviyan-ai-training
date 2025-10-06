#!/bin/bash

echo "🌱 Setting up Soil Classification Model Training Environment"
echo "=========================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv soil_training_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source soil_training_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing training dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Training environment setup complete!"
echo ""
echo "To start training:"
echo "1. Activate the environment: source soil_training_env/bin/activate"
echo "2. Run training: python train_soil_classifier.py"
echo ""
echo "🚀 Ready to train the soil classification model!"
