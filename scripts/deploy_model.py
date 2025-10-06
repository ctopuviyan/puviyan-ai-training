#!/usr/bin/env python3
"""
Model Deployment Script for Puviyan Mobile App
Copies trained model from AI repository to mobile app repository
"""

import os
import shutil
import json
from datetime import datetime
import argparse

def deploy_model(ai_repo_path, mobile_repo_path, model_name="soil_classifier_lite"):
    """
    Deploy trained model to mobile app repository
    
    Args:
        ai_repo_path: Path to puviyan-ai-training repository
        mobile_repo_path: Path to puviyan-mobile repository  
        model_name: Name of the model to deploy
    """
    
    print("🚀 Deploying Soil Detection Model")
    print("=" * 50)
    
    # Paths
    model_file = f"{model_name}.tflite"
    info_file = f"{model_name}_info.json"
    
    ai_models_dir = os.path.join(ai_repo_path, "models")
    mobile_models_dir = os.path.join(mobile_repo_path, "assets", "models")
    
    # Check source files
    source_model = os.path.join(ai_models_dir, model_file)
    source_info = os.path.join(ai_models_dir, info_file)
    
    if not os.path.exists(source_model):
        print(f"❌ Model file not found: {source_model}")
        return False
        
    # Create destination directory
    os.makedirs(mobile_models_dir, exist_ok=True)
    
    # Copy model file
    dest_model = os.path.join(mobile_models_dir, model_file)
    shutil.copy2(source_model, dest_model)
    
    model_size = os.path.getsize(dest_model) / (1024 * 1024)  # MB
    print(f"✅ Model deployed: {model_file} ({model_size:.1f} MB)")
    
    # Copy info file if exists
    if os.path.exists(source_info):
        dest_info = os.path.join(mobile_models_dir, info_file)
        shutil.copy2(source_info, dest_info)
        print(f"✅ Model info deployed: {info_file}")
    
    # Create deployment log
    deployment_log = {
        "model_name": model_name,
        "deployed_at": datetime.now().isoformat(),
        "model_size_mb": round(model_size, 2),
        "source_path": source_model,
        "destination_path": dest_model,
        "deployment_status": "success"
    }
    
    log_file = os.path.join(mobile_models_dir, "deployment_log.json")
    with open(log_file, 'w') as f:
        json.dump(deployment_log, f, indent=2)
    
    print(f"✅ Deployment log created: deployment_log.json")
    print("\n🎯 Next Steps:")
    print("1. Enable TensorFlow Lite in mobile app (uncomment in pubspec.yaml)")
    print("2. Update on-device service to use real model (remove mock mode)")
    print("3. Test the deployed model in Flutter app")
    print("4. Commit changes to mobile app repository")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Deploy trained model to mobile app")
    parser.add_argument("--ai-repo", default="../puviyan-ai-training", 
                       help="Path to AI training repository")
    parser.add_argument("--mobile-repo", default="../puviyan", 
                       help="Path to mobile app repository")
    parser.add_argument("--model", default="soil_classifier_lite",
                       help="Model name to deploy")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    ai_repo = os.path.abspath(args.ai_repo)
    mobile_repo = os.path.abspath(args.mobile_repo)
    
    print(f"AI Repository: {ai_repo}")
    print(f"Mobile Repository: {mobile_repo}")
    print(f"Model: {args.model}")
    print()
    
    # Validate paths
    if not os.path.exists(ai_repo):
        print(f"❌ AI repository not found: {ai_repo}")
        return
        
    if not os.path.exists(mobile_repo):
        print(f"❌ Mobile repository not found: {mobile_repo}")
        return
    
    # Deploy model
    success = deploy_model(ai_repo, mobile_repo, args.model)
    
    if success:
        print("\n🎉 Model deployment completed successfully!")
    else:
        print("\n❌ Model deployment failed!")

if __name__ == "__main__":
    main()
