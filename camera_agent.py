#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Computer Monitor Detection
Monitors /dev/video0 and sends alerts when a computer monitor is detected.
"""

import base64
import cv2
import json
import logging
import os
import smtplib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("camera_agent.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)


def encode_frame_optimized(frame, frame_number, timestamp, total_frames, quality=60):
    """
    Encode frame to base64 with optimized JPEG compression.
    
    Args:
        frame: OpenCV frame (BGR format)
        frame_number: Frame number in sequence
        timestamp: Timestamp of the frame
        total_frames: Total frames in sequence (for live feed, use current count)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64-encoded frame or None if encoding fails
    """
    try:
        # Encode frame as JPEG with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not success:
            logger.warning(f"Failed to encode frame {frame_number}")
            return None
        
        # Convert to base64
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64
        
    except Exception as e:
        logger.error(f"Error encoding frame {frame_number}: {e}")
        return None


def should_process_frame(current_frame, prev_frame, diff_threshold=0.80, min_time_interval=None, last_processed_time=None):
    """
    Determine if current frame should be processed based on visual difference.
    
    Args:
        current_frame: Current OpenCV frame (BGR)
        prev_frame: Previous processed frame (BGR) or None
        diff_threshold: SSIM threshold below which frames are considered different
        min_time_interval: Not used - kept for compatibility
        last_processed_time: Not used - kept for compatibility
    
    Returns:
        Tuple of (should_process: bool, similarity_score: float, reason: str)
    """
    current_time = time.time()
    
    # Always process first frame
    if prev_frame is None:
        return True, 1.0, "initial_frame"
    
    try:
        # Resize frames to smaller size for faster SSIM computation
        target_size = (320, 240)
        current_small = cv2.resize(current_frame, target_size)
        prev_small = cv2.resize(prev_frame, target_size)
        
        # Convert to grayscale for SSIM comparison
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        similarity_score = ssim(prev_gray, current_gray, data_range=255)
        
        # Process if frames are sufficiently different
        if similarity_score < diff_threshold:
            return True, similarity_score, f"scene_change (SSIM={similarity_score:.3f})"
        else:
            return False, similarity_score, f"scene_similar (SSIM={similarity_score:.3f})"
            
    except Exception as e:
        logger.warning(f"SSIM calculation failed, defaulting to process: {e}")
        # Fallback to processing frame if SSIM fails
        return True, 0.0, "ssim_fallback"


logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """Represents a monitor detection event."""
    timestamp: str
    confidence: float
    primary_label: str
    full_response: str
    image_path: Optional[str] = None


class OllamaVisionClient:
    """Client for interacting with Ollama vision models."""
    
    def __init__(self, base_url: str, model: str = "gemma3:4b", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        logger.info(f"Initialized Ollama client: {base_url}, model: {model}")
    
    def analyze_image(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        """Send image to Ollama for analysis."""
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            logger.debug(f"Sending request to {self.base_url}/api/generate")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Ollama response: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to analyze image with Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image analysis: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Ollama service."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=10)
            response.raise_for_status()
            logger.info("Ollama connection test successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False


class CameraMonitor:
    """Handles camera capture and image processing with smart frame preprocessing."""
    
    def __init__(self, device_index: int = 0, resolution: tuple = (640, 480), 
                 diff_threshold: float = 0.80, min_time_interval: float = None):
        self.device_index = device_index
        self.resolution = resolution
        self.camera = None
        
        # Frame preprocessing settings
        self.diff_threshold = diff_threshold
        self.min_time_interval = min_time_interval  # Not used anymore but kept for compatibility
        self.prev_processed_frame = None
        self.last_processed_time = None
        self.frame_count = 0
        self.processed_count = 0
        
        logger.info(f"Initializing camera monitor for device {device_index} with SSIM preprocessing (diff_threshold={diff_threshold})")
    
    def initialize_camera(self) -> bool:
        """Initialize camera connection."""
        try:
            self.camera = cv2.VideoCapture(self.device_index)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera device {self.device_index}")
                return False
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info(f"Camera initialized successfully: {self.resolution[0]}x{self.resolution[1]}")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[bytes]:
        """Capture a frame and return as JPEG bytes."""
        if not self.camera:
            logger.error("Camera not initialized")
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                return None
            
            # Convert to JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("Failed to encode frame as JPEG")
                return None
            
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def capture_frame_smart(self) -> Optional[tuple[bytes, dict]]:
        """
        Capture a frame with smart preprocessing to reduce unnecessary LLM calls.
        
        Returns:
            Tuple of (image_bytes, metadata) if frame should be processed, None otherwise
            metadata contains processing decision info
        """
        if not self.camera:
            logger.error("Camera not initialized")
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                return None
            
            self.frame_count += 1
            current_time = time.time()
            
            # Check if this frame should be processed
            should_process, similarity_score, reason = should_process_frame(
                frame, 
                self.prev_processed_frame,
                self.diff_threshold
            )
            
            metadata = {
                'frame_number': self.frame_count,
                'timestamp': current_time,
                'should_process': should_process,
                'similarity_score': similarity_score,
                'reason': reason,
                'processed_count': self.processed_count
            }
            
            if should_process:
                # Convert to JPEG bytes
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    logger.error("Failed to encode frame as JPEG")
                    return None
                
                # Update tracking variables
                self.prev_processed_frame = frame.copy()
                self.last_processed_time = current_time
                self.processed_count += 1
                
                logger.info(f"Frame {self.frame_count} selected for processing: {reason} (processed {self.processed_count}/{self.frame_count} frames)")
                
                return buffer.tobytes(), metadata
            else:
                logger.debug(f"Frame {self.frame_count} skipped: {reason}")
                return None
                
        except Exception as e:
            logger.error(f"Smart frame capture failed: {e}")
            return None
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            logger.info("Camera released")


class AlertManager:
    """Handles alert notifications via email and other channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_config = config.get("notifications", {}).get("email", {})
        logger.info("Alert manager initialized")
    
    def send_email_alert(self, event: DetectionEvent, rule: Dict[str, Any]) -> bool:
        """Send email alert for detection event."""
        if not self.email_config.get("enabled", False):
            logger.info("Email alerts disabled")
            return False
        
        try:
            # Create email message
            msg = EmailMessage()
            
            # Template substitution
            subject_template = rule.get("email", {}).get("subject", "Computer Monitor Detected")
            body_template = rule.get("email", {}).get("body", "A computer monitor was detected in the camera feed.")
            
            # Replace template variables
            template_vars = {
                "timestamp": event.timestamp,
                "confidence": f"{event.confidence:.2f}",
                "primary_label": event.primary_label,
                "full_response": event.full_response,
                "device": f"/dev/video{os.getenv('CAMERA_INDEX', '0')}"
            }
            
            subject = subject_template
            body = body_template
            for key, value in template_vars.items():
                subject = subject.replace(f"{{{{{key}}}}}", str(value))
                body = body.replace(f"{{{{{key}}}}}", str(value))
            
            msg["Subject"] = subject
            
            # Use environment variables for email credentials
            sender_email = os.getenv("EMAIL_USER") or self.email_config.get("sender_email", "noreply@zededa.com")
            sender_password = os.getenv("EMAIL_PASS") or self.email_config.get("sender_password", "")
            
            msg["From"] = os.getenv("EMAIL_FROM", sender_email)
            
            # Get recipient emails
            recipients = rule.get("actions", {}).get("email", {}).get("to", [])
            if not recipients:
                logger.error("No email recipients configured in rule")
                return False
            
            msg["To"] = ", ".join(recipients)
            msg.set_content(body)
            
            # Send email
            smtp_server = os.getenv("EMAIL_SMTP_SERVER") or self.email_config.get("smtp_server", "smtp.gmail.com")
            smtp_port = int(os.getenv("EMAIL_SMTP_PORT", self.email_config.get("smtp_port", 587)))
            
            if not sender_email or not sender_password:
                logger.error("Email credentials not configured in environment variables")
                return False
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {msg['To']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_desktop_notification(self, event: DetectionEvent) -> bool:
        """Send desktop notification (if available)."""
        try:
            import plyer
            plyer.notification.notify(
                title="ZEDEDA Camera Alert",
                message=f"Computer monitor detected at {event.timestamp}",
                timeout=10
            )
            logger.info("Desktop notification sent")
            return True
        except ImportError:
            logger.debug("plyer not available for desktop notifications")
            return False
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")
            return False


class MonitorDetectionAgent:
    """Main agent class that orchestrates camera monitoring and alert processing."""
    
    def __init__(self, config_path: str = "camera_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        ollama_url = os.getenv("OLLAMA_URL", self.config.get("ollama", {}).get("url", "http://localhost:11434"))
        vision_model = os.getenv("VISION_MODEL", self.config.get("ollama", {}).get("model", "gemma3:4b"))
        
        self.ollama_client = OllamaVisionClient(ollama_url, vision_model)
        
        camera_index = int(os.getenv("CAMERA_INDEX", self.config.get("camera", {}).get("device_index", 0)))
        camera_config = self.config.get("camera", {})
        
        # Get preprocessing settings - remove min_time_interval
        preprocessing = camera_config.get("preprocessing", {})
        diff_threshold = float(os.getenv("DIFF_THRESHOLD", preprocessing.get("diff_threshold", 0.80)))
        
        self.camera_monitor = CameraMonitor(
            camera_index, 
            diff_threshold=diff_threshold
        )
        
        self.alert_manager = AlertManager(self.config)
        
        # Runtime settings
        self.capture_interval = int(os.getenv("CAPTURE_INTERVAL", self.config.get("camera", {}).get("capture_interval", 5)))
        self.save_images = self.config.get("camera", {}).get("save_detection_images", False)
        self.save_processed_frames = self.config.get("camera", {}).get("save_processed_frames", True)
        self.images_dir = Path("detected_images")
        self.processed_frames_dir = Path("processed_frames")
        
        if self.save_images:
            self.images_dir.mkdir(exist_ok=True)
        
        if self.save_processed_frames:
            self.processed_frames_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "alerts_sent": 0,
            "start_time": datetime.now()
        }
        
        logger.info(f"Monitor Detection Agent initialized: interval={self.capture_interval}s")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "camera": {
                "device_index": 0,
                "capture_interval": 5,
                "save_detection_images": True,
                "save_processed_frames": True,
                "preprocessing": {
                    "diff_threshold": 0.80
                }
            },
            "ollama": {
                "url": "http://localhost:11434",
                "model": "gemma:4b",
                "timeout": 60
            },
            "detection": {
                "prompt": "Look carefully at this image. Is there a computer monitor, laptop screen, or any kind of display screen visible? Answer with YES if you see any type of computer screen/monitor/display, or NO if you don't see any screens. Be very specific about what you see.",
                "confidence_threshold": 0.7,
                "keywords": ["monitor", "screen", "display", "computer", "laptop", "desktop"]
            },
            "rules": [
                {
                    "id": "MONITOR_DETECTED",
                    "description": "Trigger when a computer monitor is detected",
                    "enabled": True,
                    "email": {
                        "to": ["admin@zededa.com"],
                        "subject": "Computer Monitor Detected - Camera Alert",
                        "body": """
ZEDEDA Camera Security Alert

A computer monitor/screen has been detected in the camera feed.

Details:
- Timestamp: {{timestamp}}
- Device: {{device}}
- Detection Response: {{full_response}}

This alert was generated by the ZEDEDA Camera Monitoring Agent.
"""
                    }
                }
            ],
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": ""
                },
                "desktop": {
                    "enabled": True
                }
            }
        }
    
    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Starting agent initialization...")
        
        # Test Ollama connection
        if not self.ollama_client.test_connection():
            logger.error("Failed to connect to Ollama service")
            return False
        
        # Initialize camera
        if not self.camera_monitor.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        logger.info("Agent initialization complete")
        return True
    
    def analyze_frame(self, image_data: bytes, frame_metadata: dict = None) -> Optional[DetectionEvent]:
        """Analyze a frame for monitor detection."""
        try:
            prompt = self.config.get("detection", {}).get("prompt", 
                "Look carefully at this image. Is there a computer monitor, laptop screen, or any kind of display screen visible?")
            
            # Log frame processing info
            if frame_metadata:
                logger.info(f"🔍 Analyzing frame #{frame_metadata.get('frame_number', 'unknown')} - "
                          f"Reason: {frame_metadata.get('reason', 'unknown')}, "
                          f"SSIM: {frame_metadata.get('similarity_score', 'N/A'):.3f}, "
                          f"Size: {len(image_data)} bytes")
            else:
                logger.info(f"🔍 Analyzing frame - Size: {len(image_data)} bytes")
            
            # Send to Ollama for analysis
            result = self.ollama_client.analyze_image(image_data, prompt)
            response_text = result.get("response", "").lower()
            
            # Log the actual LLM response for debugging
            full_response = result.get('response', '')
            if len(full_response) > 500:
                logger.info(f"LLM Response: {full_response[:500]}... [truncated]")
            else:
                logger.info(f"LLM Response: {full_response}")
            
            # New detection logic - check if the description contains "monitor"
            response_text = full_response.lower()
            
            detected = False
            confidence = 0.0
            
            # Check if the response contains the word "monitor" (or variations)
            monitor_keywords = [
                "monitor", "monitors", "computer monitor", "lcd monitor", "display monitor",
                "desktop monitor", "external monitor", "dual monitor", "widescreen monitor",
                "screen", "display", "computer screen", "laptop screen", "desktop display"
            ]
            
            for keyword in monitor_keywords:
                if keyword in response_text:
                    detected = True
                    confidence = 0.85
                    logger.info(f"Monitor detected - Scene description contains '{keyword}': '{full_response[:100]}...'")
                    break
            
            if not detected:
                logger.info(f"No monitor detected - Scene description: '{full_response[:100]}...'")
            
            if detected:
                event = DetectionEvent(
                    timestamp=datetime.now().isoformat(),
                    confidence=confidence,
                    primary_label="computer_monitor",
                    full_response=result.get("response", "")
                )
                
                # Save image if configured
                if self.save_images:
                    image_filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    image_path = self.images_dir / image_filename
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    event.image_path = str(image_path)
                    logger.info(f"Detection image saved: {image_path}")
                
                return event
            
            return None
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return None
    
    def process_detection(self, event: DetectionEvent) -> bool:
        """Process a detection event and send alerts."""
        logger.info(f"Processing detection: {event.primary_label} (confidence: {event.confidence:.2f})")
        
        alerts_sent = 0
        
        # Process each rule
        for rule in self.config.get("rules", []):
            if not rule.get("enabled", True):
                continue
            
            # Send email alert
            if self.alert_manager.send_email_alert(event, rule):
                alerts_sent += 1
        
        # Send desktop notification if enabled
        if self.config.get("notifications", {}).get("desktop", {}).get("enabled", False):
            if self.alert_manager.send_desktop_notification(event):
                alerts_sent += 1
        
        self.stats["alerts_sent"] += alerts_sent
        return alerts_sent > 0
    
    def run_monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting camera monitoring loop...")
        
        if not self.initialize():
            logger.error("Failed to initialize agent")
            return False
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame with smart preprocessing
                capture_result = self.camera_monitor.capture_frame_smart()
                if not capture_result:
                    # Frame was skipped due to preprocessing, wait for capture interval
                    time.sleep(self.capture_interval)
                    continue
                
                image_data, frame_metadata = capture_result
                self.stats["frames_processed"] += 1
                
                # Save processed frame if enabled
                if self.save_processed_frames:
                    processed_frame_filename = f"processed_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_metadata['frame_number']:06d}.jpg"
                    processed_frame_path = self.processed_frames_dir / processed_frame_filename
                    with open(processed_frame_path, 'wb') as f:
                        f.write(image_data)
                    logger.info(f"📸 Processed frame saved: {processed_frame_path} (Reason: {frame_metadata['reason']}, SSIM: {frame_metadata.get('similarity_score', 'N/A'):.3f})")
                
                # Log frame processing stats periodically
                if self.stats["frames_processed"] % 10 == 0:
                    processed_ratio = (frame_metadata['processed_count'] / frame_metadata['frame_number']) * 100
                    logger.info(f"📊 Frame efficiency: {frame_metadata['processed_count']}/{frame_metadata['frame_number']} processed ({processed_ratio:.1f}%), {len(os.listdir(self.processed_frames_dir))} files saved")
                
                # Analyze frame
                event = self.analyze_frame(image_data, frame_metadata)
                if event:
                    self.stats["detections"] += 1
                    logger.info(f"Monitor detected! Confidence: {event.confidence:.2f} (Frame {frame_metadata['frame_number']}, {frame_metadata['reason']})")
                    
                    # Add frame metadata to the event
                    event.full_response += f"\n\n[Frame Info: #{frame_metadata['frame_number']}, Reason: {frame_metadata['reason']}, SSIM: {frame_metadata['similarity_score']:.3f}]"
                    
                    # Process alerts
                    if self.process_detection(event):
                        logger.info("Alerts sent successfully")
                    else:
                        logger.warning("No alerts were sent")
                
                # Log statistics periodically
                if self.stats["frames_processed"] % 50 == 0:
                    runtime = datetime.now() - self.stats["start_time"]
                    total_frames = frame_metadata['frame_number']
                    processed_frames = frame_metadata['processed_count']
                    efficiency = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                    
                    logger.info(f"Statistics: {processed_frames}/{total_frames} frames processed ({efficiency:.1f}% efficiency), "
                              f"{self.stats['detections']} detections, "
                              f"{self.stats['alerts_sent']} alerts, "
                              f"runtime: {runtime}")
                
                # Sleep until next capture
                processing_time = time.time() - start_time
                sleep_time = max(0, self.capture_interval - processing_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            raise
        finally:
            self.camera_monitor.release()
            logger.info("Camera monitoring stopped")
    
    def test_setup(self) -> bool:
        """Test the complete setup without running the monitoring loop."""
        logger.info("Running setup test...")
        
        # Test Ollama connection
        if not self.ollama_client.test_connection():
            logger.error("❌ Ollama connection test failed")
            return False
        logger.info("✅ Ollama connection successful")
        
        # Test camera
        if not self.camera_monitor.initialize_camera():
            logger.error("❌ Camera initialization failed")
            return False
        logger.info("✅ Camera initialization successful")
        
        # Test frame capture
        image_data = self.camera_monitor.capture_frame()
        if not image_data:
            logger.error("❌ Frame capture test failed")
            return False
        logger.info("✅ Frame capture successful")
        
        # Test smart frame capture
        try:
            smart_result = self.camera_monitor.capture_frame_smart()
            if smart_result:
                image_data_smart, metadata = smart_result
                logger.info(f"✅ Smart frame capture successful: {metadata['reason']}")
            else:
                logger.info("✅ Smart frame capture working (frame skipped due to preprocessing)")
        except Exception as e:
            logger.warning(f"⚠️  Smart frame capture test failed (falling back to regular capture): {e}")
        
        # Test image analysis
        try:
            test_metadata = {'frame_number': 1, 'reason': 'test_frame', 'similarity_score': 1.0}
            event = self.analyze_frame(image_data, test_metadata)
            logger.info(f"✅ Image analysis successful (detection: {'Yes' if event else 'No'})")
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            return False
        
        self.camera_monitor.release()
        logger.info("✅ All tests passed!")
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZEDEDA Camera Monitor Detection Agent")
    parser.add_argument("--config", "-c", default="camera_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--test", "-t", action="store_true", 
                       help="Run setup test only")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create agent
    agent = MonitorDetectionAgent(args.config)
    
    if args.test:
        # Run test mode
        success = agent.test_setup()
        sys.exit(0 if success else 1)
    else:
        # Run monitoring loop
        try:
            agent.run_monitoring_loop()
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()