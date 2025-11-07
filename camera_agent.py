#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Computer Monitor Detection
Monitors /dev/video0 and sends alerts when a computer monitor is detected.
"""

import base64
import json
import logging
import os
import smtplib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv

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


@dataclass
class DetectionEvent:
    """Represents a monitor detection event."""
    timestamp: str
    confidence: float
    primary_label: str
    full_response: str
    image_path: Optional[str] = None
    vision_description: str = ""
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    detected: bool = False


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


class DecisionLLM:
    """Second-stage LLM for monitor detection decisions with tool calling capability."""
    
    def __init__(self, base_url: str, model: str = "llama3.2:latest", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.tools = self._define_tools()
        logger.info(f"Initialized Decision LLM: {base_url}, model: {model}")
    
    def _define_tools(self) -> List[Dict]:
        """Define available tools for the decision LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "trigger_monitor_alert",
                    "description": "Trigger an alert when a computer monitor or screen is definitively detected",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "Confidence level (0.0-1.0) that a monitor is present"
                            },
                            "monitor_type": {
                                "type": "string",
                                "description": "Type of monitor detected (e.g., 'desktop monitor', 'laptop screen', 'dual monitors')"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this is considered a positive detection"
                            }
                        },
                        "required": ["confidence", "monitor_type", "reasoning"]
                    }
                }
            }
        ]
    
    def make_detection_decision(self, vision_description: str) -> Optional[Dict[str, Any]]:
        """Use llama3.2 to decide on detection and return structured trace data."""
        try:
            classification_system_prompt = (
                "You are deciding whether a computer monitor or screen is DEFINITELY present in a description from a vision system. "
                "Respond with exactly one of the following: \n"
                "- 'DETECTION_CONFIRMED' if the description proves a qualifying screen is visible.\n"
                "- 'NO_DETECTION: <reason>' if no screen is present or you are uncertain.\n"
                "Do not use any other words."
            )

            classification_user_prompt = (
                "Vision AI description:\n"
                f"{vision_description}\n\n"
                "Is a computer monitor or other computer display definitely visible?"
            )

            classification_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": classification_system_prompt},
                    {"role": "user", "content": classification_user_prompt}
                ],
                "stream": False
            }

            logger.info(f"🤖 Decision LLM classification pass: {vision_description[:100]}...")

            classification_response = requests.post(
                f"{self.base_url}/api/chat",
                json=classification_payload,
                timeout=self.timeout
            )
            classification_response.raise_for_status()

            classification_result = classification_response.json()
            classification_message = classification_result.get("message", {})
            classification_text = classification_message.get("content", "").strip() if isinstance(classification_message.get("content"), str) else ""

            logger.debug(
                f"Decision LLM classification raw response: {json.dumps(classification_result, indent=2) if isinstance(classification_result, dict) else classification_result}"
            )

            if not classification_text:
                logger.info("Decision LLM returned empty classification response; treating as no detection")
                return {
                    "detected": False,
                    "decision_trace": {
                        "classification": "",
                        "vision_description": vision_description,
                        "error": "empty_classification_response"
                    }
                }

            if classification_text.upper().startswith("NO_DETECTION"):
                logger.info(f"✅ Decision LLM classification: {classification_text}")
                return {
                    "detected": False,
                    "decision_trace": {
                        "classification": classification_text,
                        "vision_description": vision_description
                    }
                }

            if "DETECTION_CONFIRMED" not in classification_text.upper():
                logger.warning(f"Decision LLM classification unclear: {classification_text}")
                return {
                    "detected": False,
                    "decision_trace": {
                        "classification": classification_text,
                        "vision_description": vision_description
                    }
                }

            # Positive classification – request structured tool call
            tool_system_prompt = (
                "You have already determined that a qualifying computer monitor or screen is visible. "
                "Call the function 'trigger_monitor_alert' with JSON arguments containing: confidence (0.7-1.0), monitor_type, and reasoning. "
                "Return the tool call only; no plain text."
            )

            tool_user_prompt = (
                "Vision AI description:\n"
                f"{vision_description}\n\n"
                "Provide the alert parameters now."
            )

            tool_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": tool_system_prompt},
                    {"role": "user", "content": tool_user_prompt}
                ],
                "tools": self.tools,
                "stream": False
            }

            logger.info("🤖 Decision LLM requesting structured alert via tool call...")

            tool_response = requests.post(
                f"{self.base_url}/api/chat",
                json=tool_payload,
                timeout=self.timeout
            )
            tool_response.raise_for_status()

            result = tool_response.json()
            assistant_message = result.get("message", {})
            tool_calls = assistant_message.get("tool_calls") or []

            logger.debug(
                f"Decision LLM tool-call raw response: {json.dumps(result, indent=2) if isinstance(result, dict) else result}"
            )

            for tool_call in tool_calls:
                function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                if function_payload.get("name") != "trigger_monitor_alert":
                    continue

                arguments = function_payload.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Decision LLM returned non-JSON arguments: {arguments}")
                        continue

                raw_confidence = arguments.get("confidence", 0.85)
                try:
                    confidence = float(raw_confidence)
                except (TypeError, ValueError):
                    confidence = 0.85

                monitor_type = arguments.get("monitor_type") or "monitor"
                reasoning = arguments.get("reasoning", "Monitor detected by llama3.2 decision model")

                if confidence < 0.5:
                    logger.info(
                        "Decision LLM provided tool call with low confidence %.2f; treating as no detection", confidence
                    )
                    return None

                decision_trace = {
                    "classification": classification_text,
                    "confidence": confidence,
                    "monitor_type": monitor_type,
                    "reasoning": reasoning,
                    "tool_arguments": arguments,
                    "tool_response": result,
                    "vision_description": vision_description
                }

                logger.info(f"🚨 Decision LLM triggered alert via tool call: confidence={confidence:.2f}, type={monitor_type}")
                return {
                    "detected": True,
                    "confidence": confidence,
                    "monitor_type": monitor_type,
                    "reasoning": reasoning,
                    "tool_call": True,
                    "decision_trace": decision_trace
                }

            logger.warning("Decision LLM positive classification but no tool call produced")
            return {
                "detected": False,
                "decision_trace": {
                    "classification": classification_text,
                    "vision_description": vision_description,
                    "tool_error": "positive_classification_no_tool_call"
                }
            }

        except Exception as e:
            logger.error(f"Decision LLM error: {e}")
            return {
                "detected": False,
                "decision_trace": {
                    "error": str(e),
                    "vision_description": vision_description
                }
            }
    
    def test_connection(self) -> bool:
        """Test connection to decision LLM."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with OK."}
                ],
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            assistant_message = result.get("message", {})
            content = assistant_message.get("content", "")
            return isinstance(content, str) and "OK" in content.upper()

        except Exception as e:
            logger.error(f"Decision LLM connection test failed: {e}")
            return False


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
            
            # Resolve email action configuration (supports legacy and new schema)
            actions_email = rule.get("actions", {}).get("email")
            legacy_email = rule.get("email", {})
            email_action = actions_email if actions_email is not None else legacy_email

            if not email_action:
                logger.warning(f"Rule {rule.get('id', 'unknown')} missing email configuration; skipping alert")
                return False

            if not email_action.get("enabled", legacy_email.get("enabled", True)):
                logger.info(f"Email action disabled for rule {rule.get('id', 'unknown')}")
                return False
            
            # Template substitution
            subject_template = (
                email_action.get("subject")
                or legacy_email.get("subject")
                or "Computer Monitor Detected"
            )
            body_template = (
                email_action.get("body")
                or legacy_email.get("body")
                or "A computer monitor was detected in the camera feed."
            )
            
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
            recipients = email_action.get("to") or legacy_email.get("to", [])
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
        self.last_error: Optional[str] = None
        
        # Initialize components
        ollama_url = os.getenv("OLLAMA_URL", self.config.get("ollama", {}).get("url", "http://localhost:11434"))
        vision_model = os.getenv("VISION_MODEL", self.config.get("ollama", {}).get("model", "gemma3:4b"))
        decision_model = os.getenv("DECISION_MODEL", self.config.get("ollama", {}).get("decision_model", "llama3.2:latest"))
        
        self.ollama_client = OllamaVisionClient(ollama_url, vision_model)
        self.decision_llm = DecisionLLM(ollama_url, decision_model)
        self.alert_manager = AlertManager(self.config)
        
        camera_config = self.config.get("camera", {})
        self.save_images = camera_config.get("save_detection_images", False)
        self.save_processed_frames = camera_config.get("save_processed_frames", False)
        self.images_dir = Path("detected_images")
        self.processed_frames_dir = Path("processed_frames")

        if self.save_images:
            self.images_dir.mkdir(exist_ok=True)

        if self.save_processed_frames:
            self.processed_frames_dir.mkdir(exist_ok=True)
        
        logger.info("Monitor Detection Agent initialized for web integration")
    
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
                "prompt": "Describe the image in 2-3 concise sentences."
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

    def _save_event_image(
        self,
        image_data: bytes,
        target_dir: Path,
        prefix: str,
        event_time: datetime,
        frame_metadata: Optional[dict] = None
    ) -> Optional[str]:
        """Persist an event image and return the file path."""
        try:
            target_dir.mkdir(exist_ok=True)
        except Exception as exc:
            logger.error(f"Failed to create image directory {target_dir}: {exc}")
            return None

        timestamp_str = event_time.strftime("%Y%m%d_%H%M%S_%f")
        frame_suffix = ""
        if frame_metadata and frame_metadata.get("frame_number") is not None:
            frame_suffix = f"_f{frame_metadata['frame_number']}"

        filename = f"{prefix}_{timestamp_str}{frame_suffix}.jpg"
        file_path = target_dir / filename

        try:
            with open(file_path, 'wb') as image_file:
                image_file.write(image_data)
            logger.debug(f"Event image saved: {file_path}")
            return str(file_path)
        except Exception as exc:
            logger.error(f"Failed to save event image to {file_path}: {exc}")
            return None
    
    def analyze_frame(self, image_data: bytes, frame_metadata: dict = None) -> Optional[DetectionEvent]:
        """Analyze a frame for monitor detection using two-stage LLM approach."""
        try:
            prompt = self.config.get("detection", {}).get("prompt", 
                "Describe the image in 2-3 concise sentences.")
            
            # Log frame processing info
            if frame_metadata:
                similarity_score = frame_metadata.get('similarity_score', 'N/A')
                if isinstance(similarity_score, (int, float)):
                    ssim_str = f"{similarity_score:.3f}"
                else:
                    ssim_str = str(similarity_score)
                
                logger.info(f"🔍 Analyzing frame #{frame_metadata.get('frame_number', 'unknown')} - "
                          f"Reason: {frame_metadata.get('reason', 'unknown')}, "
                          f"SSIM: {ssim_str}, "
                          f"Size: {len(image_data)} bytes")
            else:
                logger.info(f"🔍 Analyzing frame - Size: {len(image_data)} bytes")
            
            # STAGE 1: Vision LLM describes what it sees
            logger.info("🔍 Stage 1: Vision LLM analyzing image...")
            vision_result = self.ollama_client.analyze_image(image_data, prompt)
            vision_description = vision_result.get('response', '')
            
            if len(vision_description) > 500:
                logger.info(f"Vision LLM Response: {vision_description[:500]}... [truncated]")
            else:
                logger.info(f"Vision LLM Response: {vision_description}")
            
            if not vision_description:
                logger.warning("Vision LLM returned empty response")
                return None
            
            # STAGE 2: Decision LLM determines if monitor is present
            logger.info("🤖 Stage 2: Decision LLM evaluating...")
            decision_result = self.decision_llm.make_detection_decision(vision_description)
            if not isinstance(decision_result, dict):
                decision_result = {}

            decision_trace = decision_result.get("decision_trace", {})
            detected = bool(decision_result.get("detected"))
            event_time = datetime.now()
            timestamp_iso = event_time.isoformat()

            if detected:
                # Monitor detected by decision LLM
                confidence = float(decision_result.get("confidence", 0.85))
                monitor_type = decision_result.get("monitor_type", "computer_monitor")
                reasoning = decision_result.get("reasoning", "Monitor detected by decision LLM")
                
                logger.info(f"🚨 MONITOR DETECTED! Confidence: {confidence:.2f}, Type: {monitor_type}")
                logger.info(f"🎯 Detection reasoning: {reasoning}")
                
                combined_response = (
                    "Vision description:\n"
                    f"{vision_description}\n\n"
                    "Decision reasoning:\n"
                    f"{reasoning}"
                )

                detection_event = DetectionEvent(
                    timestamp=timestamp_iso,
                    confidence=confidence,
                    primary_label=monitor_type,
                    full_response=combined_response,
                    image_path=None,
                    vision_description=vision_description,
                    decision_trace=decision_trace,
                    detected=True
                )
                
                if self.save_images:
                    image_path = self._save_event_image(
                        image_data,
                        self.images_dir,
                        "detection",
                        event_time,
                        frame_metadata
                    )
                    if image_path:
                        detection_event.image_path = image_path
                
                return detection_event

            # No detection – record outcome for visibility
            classification_text = decision_trace.get("classification") or decision_result.get("classification")
            if not classification_text:
                classification_text = "NO_DETECTION"

            combined_response = (
                "Vision description:\n"
                f"{vision_description}\n\n"
                "Decision outcome:\n"
                f"{classification_text}"
            )

            logger.info("✅ No monitor detected by decision LLM")

            detection_event = DetectionEvent(
                timestamp=timestamp_iso,
                confidence=0.0,
                primary_label="no_detection",
                full_response=combined_response,
                image_path=None,
                vision_description=vision_description,
                decision_trace=decision_trace,
                detected=False
            )

            if self.save_processed_frames:
                image_path = self._save_event_image(
                    image_data,
                    self.processed_frames_dir,
                    "event",
                    event_time,
                    frame_metadata
                )
                if image_path:
                    detection_event.image_path = image_path

            return detection_event
                
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}", exc_info=True)
            self.last_error = str(e)
            return None
    
    def process_detection(self, event: DetectionEvent) -> bool:
        """Process a detection event and send alerts."""
        if not event.detected:
            logger.debug("Skipping alert processing for non-detection event")
            return False

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
        return alerts_sent > 0
    