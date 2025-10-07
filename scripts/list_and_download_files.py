#!/usr/bin/env python3
"""
List and Download Generated Model Files
======================================

This script helps you find and download all generated model files
from the Flutter-compatible training script.
"""

import os
import glob
from pathlib import Path

def find_model_files():
    """Find all generated model files"""
    print("🔍 Searching for generated model files...")
    print("=" * 50)
    
    # Look for flutter_compatible directories
    flutter_dirs = glob.glob("flutter_compatible_*")
    
    if not flutter_dirs:
        print("❌ No flutter_compatible_* directories found")
        print("💡 Make sure you're in the directory where training was run")
        return
    
    # Sort by creation time (newest first)
    flutter_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for i, dir_path in enumerate(flutter_dirs):
        print(f"\n📁 Directory {i+1}: {dir_path}")
        print("-" * 30)
        
        # List all files in the directory
        files_found = []
        for file_path in Path(dir_path).glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                files_found.append((str(file_path), size_mb))
                print(f"   📄 {file_path.name} ({size_mb:.2f} MB)")
        
        if not files_found:
            print("   ⚠️ No files found in this directory")
        else:
            print(f"   ✅ Found {len(files_found)} files")
    
    return flutter_dirs

def download_files_colab(directory):
    """Download files using Google Colab files.download()"""
    try:
        from google.colab import files
        
        print(f"\n📥 Downloading files from {directory}...")
        print("=" * 50)
        
        # Find all files in the directory
        for file_path in Path(directory).glob("*"):
            if file_path.is_file():
                try:
                    print(f"📱 Downloading {file_path.name}...")
                    files.download(str(file_path))
                    print(f"✅ Downloaded: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to download {file_path.name}: {e}")
        
        print("\n✅ Download process completed!")
        
    except ImportError:
        print("⚠️ Not in Google Colab environment")
        print("💡 Files are available in the directory for manual download")
    except Exception as e:
        print(f"❌ Download failed: {e}")

def copy_to_current_directory(directory):
    """Copy all files to current directory for easier access"""
    import shutil
    
    print(f"\n📁 Copying files from {directory} to current directory...")
    print("=" * 50)
    
    copied_files = []
    for file_path in Path(directory).glob("*"):
        if file_path.is_file():
            try:
                dest_path = Path(file_path.name)
                shutil.copy2(file_path, dest_path)
                print(f"✅ Copied: {file_path.name}")
                copied_files.append(str(dest_path))
            except Exception as e:
                print(f"❌ Failed to copy {file_path.name}: {e}")
    
    return copied_files

def main():
    """Main function"""
    print("🌱 Flutter Model Files Manager")
    print("=" * 50)
    
    # Find all model directories
    directories = find_model_files()
    
    if not directories:
        return
    
    # Use the most recent directory
    latest_dir = directories[0]
    print(f"\n🎯 Using latest directory: {latest_dir}")
    
    # Copy files to current directory
    copied_files = copy_to_current_directory(latest_dir)
    
    # Try to download in Colab
    download_files_colab(".")  # Download from current directory
    
    print(f"\n🎉 Process Complete!")
    print("=" * 50)
    print("📱 Your Flutter-compatible model files:")
    for file_path in copied_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.2f} MB)")
    
    print("\n💡 Manual download commands (if auto-download fails):")
    for file_path in copied_files:
        print(f"   files.download('{file_path}')")

if __name__ == "__main__":
    main()
