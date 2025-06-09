#!/usr/bin/env python3
"""
Setup script for Pune Flat Price Predictor
This script sets up the complete ML project with data generation, model training, and Streamlit app.
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually using: pip install -r requirements.txt")
        return False
    return True

def generate_data():
    """Generate the Pune real estate dataset"""
    print("🏠 Generating Pune real estate dataset...")
    try:
        subprocess.check_call([sys.executable, "generate_pune_data.py"])
        print("✅ Dataset generated successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to generate dataset.")
        return False
    return True

def train_model():
    """Train the ML models"""
    print("🤖 Training machine learning models...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Models trained successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to train models.")
        return False
    return True

def run_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Launching Streamlit application...")
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit app.")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🏠 PUNE FLAT PRICE PREDICTOR - SETUP")
    print("=" * 60)
    
    # Check if all files exist
    required_files = [
        "requirements.txt",
        "generate_pune_data.py", 
        "train_model.py",
        "app.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all project files are in the current directory.")
        return
    
    print("📋 Setup Steps:")
    print("1. Install requirements")
    print("2. Generate dataset") 
    print("3. Train ML models")
    print("4. Launch Streamlit app")
    print()
    
    # Step 1: Install requirements
    if not install_requirements():
        return
    
    # Step 2: Generate dataset
    if not generate_data():
        return
    
    # Step 3: Train models
    if not train_model():
        return
    
    print("\n✅ Setup completed successfully!")
    print("🎉 Ready to launch the application!")
    print()
    
    # Step 4: Launch app
    launch = input("Launch Streamlit app now? (y/n): ").lower().strip()
    if launch in ['y', 'yes', '']:
        run_streamlit()
    else:
        print("🚀 To launch the app later, run: streamlit run app.py")

if __name__ == "__main__":
    main()
