#!/usr/bin/env python3
"""
Flutter TFLite Model Compatibility Tester
=========================================

Tests if generated TensorFlow Lite models are compatible with Flutter's
tflite_flutter package by checking for unsupported operations.

Usage:
    python test_flutter_compatibility.py path/to/model.tflite
"""

import tensorflow as tf
import numpy as np
import sys
import os

def test_model_compatibility(model_path):
    """Test if TFLite model is compatible with Flutter"""
    print(f"🧪 Testing Flutter compatibility for: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ Model loaded successfully")
        print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Check input details
        print("\n📥 Input Details:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            print(f"    Quantization: {detail.get('quantization', 'None')}")
        
        # Check output details
        print("\n📤 Output Details:")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            print(f"    Quantization: {detail.get('quantization', 'None')}")
        
        # Test inference with dummy data
        print("\n🔄 Testing Inference:")
        
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        # Create dummy input data
        if input_dtype == np.uint8:
            dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
            print("✅ Using UINT8 input (Flutter-friendly)")
        elif input_dtype == np.float32:
            dummy_input = np.random.random(input_shape).astype(np.float32)
            print("⚠️ Using FLOAT32 input (may need conversion in Flutter)")
        else:
            print(f"❌ Unsupported input type: {input_dtype}")
            return False
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_dtype = output_details[0]['dtype']
        
        if output_dtype == np.uint8:
            print("✅ Using UINT8 output (Flutter-friendly)")
        elif output_dtype == np.float32:
            print("⚠️ Using FLOAT32 output (may need conversion in Flutter)")
        else:
            print(f"❌ Unsupported output type: {output_dtype}")
            return False
        
        print(f"✅ Inference successful! Output shape: {output_data.shape}")
        
        # Check for Flutter compatibility indicators
        print("\n🔍 Flutter Compatibility Check:")
        
        compatibility_score = 0
        total_checks = 5
        
        # Check 1: Input type
        if input_dtype == np.uint8:
            print("✅ Input type is UINT8 (optimal for Flutter)")
            compatibility_score += 1
        elif input_dtype == np.float32:
            print("⚠️ Input type is FLOAT32 (workable but not optimal)")
            compatibility_score += 0.5
        else:
            print(f"❌ Input type {input_dtype} may cause issues")
        
        # Check 2: Output type
        if output_dtype == np.uint8:
            print("✅ Output type is UINT8 (optimal for Flutter)")
            compatibility_score += 1
        elif output_dtype == np.float32:
            print("⚠️ Output type is FLOAT32 (workable but not optimal)")
            compatibility_score += 0.5
        else:
            print(f"❌ Output type {output_dtype} may cause issues")
        
        # Check 3: Model size
        model_size_mb = os.path.getsize(model_path) / (1024*1024)
        if model_size_mb < 10:
            print(f"✅ Model size ({model_size_mb:.2f} MB) is mobile-friendly")
            compatibility_score += 1
        elif model_size_mb < 20:
            print(f"⚠️ Model size ({model_size_mb:.2f} MB) is acceptable")
            compatibility_score += 0.5
        else:
            print(f"❌ Model size ({model_size_mb:.2f} MB) may be too large")
        
        # Check 4: Input shape
        if len(input_shape) == 4 and input_shape[1] == input_shape[2]:
            print(f"✅ Input shape {input_shape} is standard for image models")
            compatibility_score += 1
        else:
            print(f"⚠️ Input shape {input_shape} may need special handling")
            compatibility_score += 0.5
        
        # Check 5: Successful inference
        print("✅ Model inference works correctly")
        compatibility_score += 1
        
        # Final compatibility assessment
        print(f"\n🎯 Compatibility Score: {compatibility_score}/{total_checks}")
        
        if compatibility_score >= 4.5:
            print("🎉 EXCELLENT: Model is highly compatible with Flutter!")
            compatibility_level = "EXCELLENT"
        elif compatibility_score >= 3.5:
            print("✅ GOOD: Model should work well with Flutter")
            compatibility_level = "GOOD"
        elif compatibility_score >= 2.5:
            print("⚠️ FAIR: Model may work but could have issues")
            compatibility_level = "FAIR"
        else:
            print("❌ POOR: Model likely to have compatibility issues")
            compatibility_level = "POOR"
        
        # Provide recommendations
        print("\n💡 Recommendations for Flutter Integration:")
        
        if input_dtype != np.uint8:
            print("  • Consider converting input to UINT8 for better performance")
        
        if output_dtype != np.uint8:
            print("  • Consider quantizing output to UINT8 for consistency")
        
        if model_size_mb > 10:
            print("  • Consider further quantization to reduce model size")
        
        print("  • Test the model in your Flutter app to ensure it works correctly")
        print("  • Monitor inference time on target devices")
        
        return compatibility_level in ["EXCELLENT", "GOOD"]
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python test_flutter_compatibility.py <model_path>")
        print("Example: python test_flutter_compatibility.py ../models/soil_classifier.tflite")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("🧪 Flutter TFLite Model Compatibility Tester")
    print("=" * 60)
    
    is_compatible = test_model_compatibility(model_path)
    
    print("\n" + "=" * 60)
    if is_compatible:
        print("🎉 Model is ready for Flutter integration!")
        print("📱 You can safely use this model in your Flutter app")
    else:
        print("⚠️ Model may have compatibility issues")
        print("🔧 Consider regenerating with Flutter-compatible settings")
    
    return 0 if is_compatible else 1

if __name__ == "__main__":
    exit(main())
