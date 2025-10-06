# 📓 Puviyan Training Notebooks

## 🚀 **Recommended: Use v3.0.0 Notebook**

### **📱 Soil_Detection_Training_v3.ipynb** ⭐ **LATEST**
- **✅ Direct GitHub download** - No manual file upload needed
- **✅ Real dataset support** - Upload ZIP files with soil images
- **✅ Prevents incremental filenames** (.py.1, .py.2 issues)
- **✅ Enhanced troubleshooting** tools
- **✅ v3.0.0 features** - Latest training script
- **✅ Complete workflow** - Setup to deployment

### **📱 Soil_Detection_Training.ipynb** ❌ **OUTDATED**
- **❌ Manual file upload** required
- **❌ No real dataset support**
- **❌ Incremental filename issues**
- **❌ Limited troubleshooting**

## 🎯 **Quick Start (Copy-Paste in Colab):**

### **Option 1: Use v3.0.0 Notebook (Recommended)**
1. **Open in Colab**: [Soil_Detection_Training_v3.ipynb](https://colab.research.google.com/github/ctopuviyan/puviyan-ai-training/blob/main/notebooks/Soil_Detection_Training_v3.ipynb)
2. **Enable GPU**: Runtime > Change runtime type > GPU
3. **Run all cells** - Everything is automated!

### **Option 2: Manual Setup (Advanced Users)**
```python
# 🧹 Complete cleanup and download
import os
for f in os.listdir('.'):
    if f.startswith('train_soil_classifier.py'):
        os.remove(f)

# 📥 Download latest script
!wget -O train_soil_classifier.py https://raw.githubusercontent.com/ctopuviyan/puviyan-ai-training/main/scripts/train_soil_classifier.py

# 🚀 Run training
!python train_soil_classifier.py
```

## 📊 **Features Comparison:**

| Feature | v3.0.0 Notebook | Old Notebook |
|---------|----------------|--------------|
| GitHub Download | ✅ Automatic | ❌ Manual Upload |
| Real Dataset Support | ✅ ZIP Upload | ❌ Synthetic Only |
| Incremental Filename Fix | ✅ Prevented | ❌ Common Issue |
| Version Verification | ✅ Built-in | ❌ Manual Check |
| Troubleshooting Tools | ✅ Comprehensive | ❌ Basic |
| Dataset Choice | ✅ Interactive | ❌ Fixed |
| Mobile Deployment Guide | ✅ Complete | ❌ Basic |

## 🎯 **Expected Training Output:**

```
🌱 Puviyan Soil Detection Training v3.0.0
============================================================
✅ Running in Google Colab
✅ TensorFlow: 2.x.x
✅ GPU Available: 1 device(s)
============================================================

📊 Dataset Options:
1. Upload real soil images (ZIP file)
2. Use synthetic dataset generation

Choose dataset type (1 for real, 2 for synthetic): 
```

## 🔗 **Links:**

- **📓 v3.0.0 Notebook**: [Soil_Detection_Training_v3.ipynb](./Soil_Detection_Training_v3.ipynb)
- **📚 Setup Instructions**: [Colab_Setup_Instructions.md](./Colab_Setup_Instructions.md)
- **🐍 Training Script**: [train_soil_classifier.py](../scripts/train_soil_classifier.py)
- **📱 Mobile App**: [Puviyan Flutter App](https://github.com/ctopuviyan/puviyan)

---

**🌱 Happy Training! Use the v3.0.0 notebook for the best experience! 🇮🇳**
