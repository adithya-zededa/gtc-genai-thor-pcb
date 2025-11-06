#!/usr/bin/env python3
"""
ZEDEDA Camera Agent Demo
Demonstrates camera monitoring and computer monitor detection
"""

import cv2
import time
import os
from datetime import datetime

def demo_camera_capture():
    """Demo camera capture functionality"""
    print("🎥 ZEDEDA Camera Agent Demo")
    print("=" * 50)
    
    # Initialize camera
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    print(f"📷 Initializing camera /dev/video{camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Failed to open camera /dev/video{camera_index}")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera initialized successfully!")
    print("\nDemo will capture frames every 2 seconds for 30 seconds")
    print("Position a computer monitor in view to simulate detection")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 30:  # Run for 30 seconds
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to capture frame")
                continue
            
            frame_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save frame for analysis
            filename = f"demo_frame_{frame_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            
            print(f"📸 Frame {frame_count}: Captured {filename} at {timestamp}")
            
            # Simulate detection analysis
            print(f"🤖 [SIMULATED] Analyzing frame for computer monitors...")
            print(f"🔍 [SIMULATED] Would send frame to Ollama llava model")
            print(f"📧 [SIMULATED] Would check detection rules and send alerts if triggered")
            print()
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    finally:
        cap.release()
        print(f"\n✅ Demo complete! Captured {frame_count} frames")
        print("🔧 When Ollama is ready, the real agent will:")
        print("   1. Analyze each frame with AI vision model")
        print("   2. Detect computer monitors/screens") 
        print("   3. Send email alerts when monitors are found")
        print("   4. Log all detections with timestamps")

if __name__ == "__main__":
    demo_camera_capture()