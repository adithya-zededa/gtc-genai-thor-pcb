#!/usr/bin/env python3
"""
Quick test script for ZEDEDA Camera Agent
Tests camera access, Ollama connectivity, and basic functionality
"""

import cv2
import os
import sys
import requests
import time
from pathlib import Path

def test_camera_access():
    """Test if camera is accessible"""
    print("🔍 Testing camera access...")
    
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Failed to open camera /dev/video{camera_index}")
            return False
        
        # Try to capture a frame
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture frame from camera /dev/video{camera_index}")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"✅ Camera /dev/video{camera_index} is working - Resolution: {width}x{height}")
        
        # Save a test image
        test_image_path = "camera_test.jpg"
        cv2.imwrite(test_image_path, frame)
        print(f"✅ Test image saved: {test_image_path}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama service connectivity"""
    print("\n🔍 Testing Ollama connection...")
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    try:
        # Test version endpoint
        response = requests.get(f"{ollama_url}/api/version", timeout=10)
        response.raise_for_status()
        
        version_info = response.json()
        print(f"✅ Ollama is running - Version: {version_info.get('version', 'unknown')}")
        
        # Test models endpoint
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            model_names = [model.get('name', 'unknown') for model in models.get('models', [])]
            print(f"✅ Available models: {', '.join(model_names) if model_names else 'none'}")
            
            # Check for vision models
            vision_models = [name for name in model_names if 'llava' in name.lower()]
            if vision_models:
                print(f"✅ Vision models found: {', '.join(vision_models)}")
            else:
                print(f"⚠️  No LLaVA vision models found")
                print(f"   Install with: ollama pull llava:7b")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama connection failed: {e}")
        print(f"   Make sure Ollama is running at {ollama_url}")
        return False

def test_vision_analysis():
    """Test vision analysis with a sample image"""
    print("\n🔍 Testing vision analysis...")
    
    # Check if test image exists
    test_image_path = "camera_test.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    vision_model = os.getenv("VISION_MODEL", "llava:7b")
    
    try:
        import base64
        
        # Read and encode image
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare request
        prompt = "Look at this image. Can you see any computer monitors, laptop screens, or displays? Describe what you see."
        
        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False
        }
        
        print(f"📤 Sending image to {vision_model}...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        ai_response = result.get("response", "No response")
        
        print(f"✅ Vision analysis successful!")
        print(f"🤖 AI Response: {ai_response}")
        
        # Check for monitor-related keywords
        monitor_keywords = ["monitor", "screen", "display", "computer", "laptop"]
        found_keywords = [kw for kw in monitor_keywords if kw.lower() in ai_response.lower()]
        
        if found_keywords:
            print(f"🎯 Monitor-related keywords found: {', '.join(found_keywords)}")
        else:
            print(f"ℹ️  No monitor-related keywords detected in response")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision analysis failed: {e}")
        return False

def test_config_file():
    """Test configuration file"""
    print("\n🔍 Testing configuration file...")
    
    config_path = "camera_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration file loaded successfully")
        
        # Check key sections
        required_sections = ["camera", "ollama", "detection", "rules", "notifications"]
        for section in required_sections:
            if section in config:
                print(f"✅ Configuration section '{section}' found")
            else:
                print(f"⚠️  Configuration section '{section}' missing")
        
        # Check email configuration
        email_config = config.get("notifications", {}).get("email", {})
        if email_config.get("enabled"):
            if email_config.get("sender_email") and email_config.get("sender_password"):
                print(f"✅ Email configuration appears complete")
            else:
                print(f"⚠️  Email enabled but credentials missing")
        else:
            print(f"ℹ️  Email notifications disabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ZEDEDA Camera Agent - Quick Test")
    print("=" * 50)
    
    # Load environment variables if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Environment variables loaded from {env_file}")
    
    tests = [
        ("Camera Access", test_camera_access),
        ("Ollama Connection", test_ollama_connection),
        ("Configuration File", test_config_file),
        ("Vision Analysis", test_vision_analysis),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Camera agent is ready to run.")
        print("\nNext steps:")
        print("1. Update email credentials in .env file")
        print("2. Run: ./start_camera_agent.sh start")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before running the agent.")
        return 1

if __name__ == "__main__":
    sys.exit(main())