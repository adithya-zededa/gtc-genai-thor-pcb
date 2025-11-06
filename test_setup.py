#!/usr/bin/env python3
"""
Quick test script to verify the agent setup without actually running the full agent.
"""

import sys
import subprocess

def check_dependency(module_name, package_name=None):
    """Check if a Python module is available."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        print(f"   Install with: pip3 install {package_name}")
        return False

def check_ollama():
    """Check if Ollama service is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            
            # Check for vision models
            data = response.json()
            models = data.get('models', [])
            vision_models = [m for m in models if 'llava' in m.get('name', '').lower()]
            
            if vision_models:
                print(f"✅ Vision models found: {len(vision_models)}")
                for model in vision_models[:3]:  # Show first 3
                    print(f"   - {model.get('name')}")
            else:
                print("⚠️  No LLaVA vision models found")
                print("   Install with: ollama pull llava")
            
            return True
    except Exception as e:
        print(f"❌ Ollama service is NOT running: {e}")
        print("   Start with: ollama serve")
        return False

def check_camera():
    """Check if camera is accessible."""
    try:
        import cv2
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            print("✅ Webcam is accessible")
            cam.release()
            return True
        else:
            print("❌ Webcam NOT accessible")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def check_config_files():
    """Check if configuration files exist."""
    import os
    
    files_to_check = {
        'agent.py': 'Main agent script',
        'agent_enhanced.py': 'Enhanced agent with email/Slack',
        'requirements.txt': 'Python dependencies',
        'alert_rules.json': 'Alert rules configuration',
        '.env.example': 'Environment configuration template',
        'AGENT_README.md': 'Agent documentation',
        'QUICKSTART.md': 'Quick start guide',
    }
    
    all_exist = True
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - MISSING!")
            all_exist = False
    
    return all_exist

def main():
    print("="*70)
    print("🔍 ZEDEDA Security Agent - Setup Verification")
    print("="*70)
    print()
    
    # Check configuration files
    print("📁 Checking configuration files...")
    print("-" * 70)
    config_ok = check_config_files()
    print()
    
    # Check Python dependencies
    print("📦 Checking Python dependencies...")
    print("-" * 70)
    deps_ok = True
    deps_ok &= check_dependency('cv2', 'opencv-python')
    deps_ok &= check_dependency('requests')
    deps_ok &= check_dependency('dotenv', 'python-dotenv')
    
    # Optional dependencies
    print("\n🔔 Checking optional dependencies...")
    print("-" * 70)
    check_dependency('plyer')  # Desktop notifications
    print()
    
    # Check Ollama service
    print("🤖 Checking Ollama service...")
    print("-" * 70)
    ollama_ok = check_ollama()
    print()
    
    # Check camera
    print("📷 Checking webcam...")
    print("-" * 70)
    camera_ok = check_camera()
    print()
    
    # Final summary
    print("="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    all_checks = [
        ("Configuration files", config_ok),
        ("Python dependencies", deps_ok),
        ("Ollama service", ollama_ok),
        ("Webcam access", camera_ok),
    ]
    
    for check_name, result in all_checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    print()
    
    if all([result for _, result in all_checks]):
        print("🎉 All checks passed! You're ready to run the agent.")
        print()
        print("Run the agent with:")
        print("  python3 agent.py                # Basic agent")
        print("  python3 agent_enhanced.py       # Enhanced agent")
        print("  ./start_agent.sh                # Automated setup")
        return 0
    else:
        print("⚠️  Some checks failed. Please resolve the issues above.")
        print()
        print("Quick fixes:")
        if not deps_ok:
            print("  pip3 install -r requirements.txt")
        if not ollama_ok:
            print("  ollama serve")
            print("  ollama pull llava")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(1)
