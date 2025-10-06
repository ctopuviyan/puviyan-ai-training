# 🌱 Puviyan AI Training - Soil Detection Model

AI training pipeline for Indian soil type classification using TensorFlow and computer vision.

## 🎯 **Overview**

This repository contains the machine learning pipeline for training soil detection models used in the Puviyan mobile app. The model classifies soil into 8 major Indian soil types to help users select appropriate trees for planting.

## 🌾 **Soil Types Supported**

| Index | Soil Type | Characteristics | Best Trees |
|-------|-----------|-----------------|------------|
| 0 | **Alluvial Soil** | Fertile river deposits | Most tree species |
| 1 | **Black Soil** | Clay-rich, moisture retentive | Neem, Banyan, Cotton trees |
| 2 | **Red Soil** | Iron-rich, well-draining | Cashew, Coconut, Areca nut |
| 3 | **Laterite Soil** | Iron/aluminum rich | Coconut, Cashew, Mango |
| 4 | **Desert Soil** | Arid, low organic content | Acacia, Prosopis |
| 5 | **Saline/Alkaline** | High salt content | Casuarina, Prosopis |
| 6 | **Peaty/Marshy** | Waterlogged, organic-rich | Mangroves, Willows, Bamboo |
| 7 | **Forest/Hill** | Rich humus, forest soil | Native forest species |

## 📁 **Repository Structure**

```
puviyan-ai-training/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📁 scripts/                     # Training scripts
│   ├── train_soil_classifier.py    # Main training pipeline
│   ├── create_basic_model.py       # Model architecture utilities
│   ├── create_simple_model.py      # Alternative architectures
│   └── setup_training.sh           # Environment setup
├── 📁 notebooks/                   # Jupyter notebooks for analysis
├── 📁 datasets/                    # Training data (not in git)
├── 📁 models/                      # Trained models output
└── 📁 docs/                        # Documentation
```

## 🚀 **Quick Start**

### **Option 1: Google Colab (Recommended)**

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Upload training script**: Upload `scripts/train_soil_classifier.py`
3. **Enable GPU**: Runtime → Change runtime type → GPU
4. **Run training**:
   ```python
   !python train_soil_classifier.py
   ```
5. **Download model**: The trained `.tflite` file will be generated

### **Option 2: Local Training**

```bash
# Clone repository
git clone <repository-url>
cd puviyan-ai-training

# Setup environment
chmod +x scripts/setup_training.sh
./scripts/setup_training.sh

# Install dependencies
pip install -r requirements.txt

# Run training (requires GPU for reasonable speed)
python scripts/train_soil_classifier.py
```

## 📊 **Model Specifications**

- **Architecture**: MobileNetV2 + Custom Classification Head
- **Input Size**: 224×224×3 (RGB images)
- **Output**: 8 classes (Indian soil types)
- **Model Size**: ~5-10MB (TensorFlow Lite)
- **Inference Time**: ~150ms on mobile devices
- **Accuracy Target**: >85% on validation set

## 🔧 **Training Configuration**

```python
# Default training parameters
INPUT_SIZE = 224
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

## 📚 **Usage**

### **Training with Real Data**

1. **Prepare Dataset**:
   ```
   datasets/
   ├── alluvial/
   │   ├── img001.jpg
   │   └── img002.jpg
   ├── black/
   ├── red/
   ├── laterite/
   ├── desert/
   ├── saline/
   ├── peaty/
   └── forest/
   ```

2. **Run Training**:
   ```bash
   python scripts/train_soil_classifier.py --data_dir datasets/
   ```

3. **Deploy Model**:
   ```bash
   # Copy trained model to mobile app
   cp models/soil_classifier_lite.tflite ../puviyan-mobile/assets/models/
   ```

### **Training with Synthetic Data**

For testing and development:

```bash
python scripts/train_soil_classifier.py --synthetic_data
```

## 🎯 **Model Performance**

### **Evaluation Metrics**
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **Confusion Matrix**: Classification errors analysis
- **Inference Time**: Mobile performance metrics

### **Expected Results**
```
Training Accuracy: ~95%
Validation Accuracy: ~87%
Model Size: 8.5MB
Inference Time: 150ms (mobile)
```

## 🔄 **Integration with Mobile App**

### **Model Deployment**
1. Train model using this repository
2. Copy `.tflite` file to mobile app: `puviyan-mobile/assets/models/`
3. Mobile app automatically uses new model

### **Model Versioning**
- Models are versioned by timestamp
- Mobile app can check for model updates
- Backward compatibility maintained

## 🛠️ **Development**

### **Adding New Soil Types**
1. Update `NUM_CLASSES` in training script
2. Add new soil type to `SOIL_LABELS` dictionary
3. Prepare training data for new soil type
4. Retrain model
5. Update mobile app soil type enum

### **Improving Model**
- **More Data**: Collect more diverse soil images
- **Data Augmentation**: Rotation, brightness, contrast adjustments
- **Architecture**: Experiment with different base models
- **Hyperparameters**: Tune learning rate, batch size, epochs

## 📋 **Requirements**

### **Python Dependencies**
```
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
pillow>=8.3.0
scikit-learn>=1.0.0
```

### **Hardware Requirements**
- **Training**: GPU recommended (Google Colab free tier sufficient)
- **Storage**: 10GB+ for datasets and models
- **RAM**: 8GB+ recommended

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-soil-type`
3. Make changes and test
4. Submit pull request

## 📄 **License**

This project is part of the Puviyan ecosystem for environmental conservation and tree planting initiatives.

## 🔗 **Related Repositories**

- **Mobile App**: `puviyan-mobile` - Flutter app using the trained models
- **Backend API**: `puviyan-backend` - Server-side services
- **Documentation**: `puviyan-docs` - Complete project documentation

## 📞 **Support**

For questions about model training or AI-related issues:
- Create an issue in this repository
- Contact the AI team
- Check documentation in `docs/` folder

---

**🌱 Building AI for a greener future! 🌍**
