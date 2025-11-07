#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Web Frontend
Modern Flask web interface for camera monitoring and configuration
"""

import os
import json
import yaml
import base64
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response, send_file
from flask_socketio import SocketIO, emit
import threading
import time
from camera_agent import MonitorDetectionAgent
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import requests
import csv
import io

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
camera_agent = None
DATA_DIR = Path('.')
DETECTED_IMAGES_DIR = DATA_DIR / 'detected_images'
PROCESSED_FRAMES_DIR = DATA_DIR / 'processed_frames'

class WebCameraAgent:
    """Web-integrated camera agent with real-time updates"""
    
    def __init__(self):
        self.agent = None
        self.is_monitoring = False
        self.last_frame = None
        self.last_error = None
        self._thread: Optional[threading.Thread] = None
        self.camera = None  # Web app owns the camera
        self.last_processed_time = 0
        self.capture_interval = 5  # seconds between agent processing
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'uptime_start': datetime.now().isoformat()
        }
    
    def initialize(self):
        """Initialize the camera agent (without camera - web app handles that)"""
        try:
            self.agent = MonitorDetectionAgent()
            # Don't initialize camera in agent - web app will handle it
            # Just test Ollama connection
            if not self.agent.ollama_client.test_connection():
                self.last_error = f"Failed to connect to Ollama service at {self.agent.ollama_client.base_url}"
                return False
            
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"Failed to initialize camera agent: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera for web app streaming"""
        try:
            if self.camera and self.camera.isOpened():
                return True
            
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.last_error = "Failed to open camera"
                return False
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.is_monitoring:
            return False
        if not self.agent:
            if not self.initialize():
                return False
        
        # Initialize camera for web app
        if not self.initialize_camera():
            return False

        self.is_monitoring = True
        thread = threading.Thread(target=self._monitoring_loop)
        thread.daemon = True
        thread.start()
        self._thread = thread
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def _monitoring_loop(self):
        """Main monitoring loop - web app captures, agent analyzes periodically"""
        prev_frame_for_ssim = None
        frame_count = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Ensure camera is available
                if not self.camera or not self.camera.isOpened():
                    if not self.initialize_camera():
                        self.last_error = "Camera connection lost"
                        time.sleep(2)
                        continue
                    self.last_error = None
                
                # Capture frame from web app's camera
                ret, frame = self.camera.read()
                if not ret:
                    self.last_error = "Failed to capture frame"
                    time.sleep(1)
                    continue
                
                frame_count += 1
                self.stats['total_frames'] += 1
                
                # Encode frame for storage/display
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    image_data = buffer.tobytes()
                    
                    # Store latest frame for web display
                    self.last_frame = {
                        'image_b64': base64.b64encode(image_data).decode('utf-8'),
                        'timestamp': datetime.now().isoformat(),
                        'frame_number': frame_count
                    }
                    
                    # Emit frame update to web clients
                    socketio.emit('frame_update', {
                        'frame': self.last_frame,
                        'stats': self._serialize_stats()
                    })
                
                # Check if enough time has passed to send frame to agent
                if current_time - self.last_processed_time >= self.capture_interval:
                    self.last_processed_time = current_time
                    
                    # Check if frame is different enough using SSIM (agent's logic)
                    should_process = True
                    reason = "periodic_check"
                    
                    if prev_frame_for_ssim is not None:
                        try:
                            from skimage.metrics import structural_similarity as ssim
                            import numpy as np
                            
                            # Resize for faster SSIM
                            target_size = (320, 240)
                            current_small = cv2.resize(frame, target_size)
                            prev_small = cv2.resize(prev_frame_for_ssim, target_size)
                            
                            current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
                            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
                            
                            similarity = ssim(prev_gray, current_gray, data_range=255)
                            
                            if similarity < 0.80:  # Scene changed
                                should_process = True
                                reason = f"scene_change (SSIM={similarity:.3f})"
                            else:
                                should_process = False
                                reason = f"scene_similar (SSIM={similarity:.3f})"
                        except Exception as e:
                            print(f"SSIM check failed: {e}")
                            should_process = True
                            reason = "ssim_fallback"
                    
                    if should_process:
                        self.stats['processed_frames'] += 1
                        prev_frame_for_ssim = frame.copy()
                        
                        # Create metadata
                        frame_metadata = {
                            'frame_number': frame_count,
                            'timestamp': current_time,
                            'reason': reason,
                            'processed_count': self.stats['processed_frames']
                        }
                        
                        print(f"🔍 Sending frame {frame_count} to agent for analysis: {reason}")
                        
                        # Send to agent for LLM analysis
                        event = self.agent.analyze_frame(image_data, frame_metadata)
                        
                        if event:
                            self.stats['detections'] += 1
                            print(f"🔔 Detection! Confidence: {event.confidence:.2f}")
                            
                            # Process alerts
                            if self.agent.process_detection(event):
                                self.stats['alerts_sent'] += 1
                            
                            self._record_detection(event, frame_metadata)
                            
                            # Emit detection event
                            socketio.emit('detection_event', {
                                'event': {
                                    'timestamp': event.timestamp,
                                    'confidence': event.confidence,
                                    'response': event.full_response[:200] + '...' if len(event.full_response) > 200 else event.full_response
                                },
                                'stats': self._serialize_stats()
                            })
                        else:
                            print(f"   No detection in frame {frame_count}")
                
                # Small delay to control frame rate
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                self.last_error = str(e)
                time.sleep(2)

    def _serialize_stats(self):
        """Return stats dict with JSON-serializable values"""
        stats_copy = self.stats.copy()
        uptime_start = stats_copy.get('uptime_start')
        if isinstance(uptime_start, datetime):
            stats_copy['uptime_start'] = uptime_start.isoformat()
        stats_copy['last_update'] = datetime.now().isoformat()
        return stats_copy

    def _record_detection(self, event, frame_metadata):
        """Persist detection event to the database for UI visibility"""
        try:
            conn = get_db_connection()
            conn.execute(
                '''
                INSERT INTO detection_logs (timestamp, confidence, response, image_path, frame_number, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    event.timestamp,
                    event.confidence,
                    event.full_response,
                    event.image_path or '',
                    frame_metadata.get('frame_number'),
                    frame_metadata.get('reason')
                )
            )
            conn.commit()
        except Exception as e:
            print(f"Failed to record detection: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

# Initialize database
def init_db():
    """Initialize SQLite database for user management and logs"""
    conn = sqlite3.connect('camera_agent.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Detection logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            response TEXT,
            image_path TEXT,
            frame_number INTEGER,
            reason TEXT
        )
    ''')
    
    # Configuration history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config_type TEXT,
            changes TEXT,
            user_email TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect('camera_agent.db')
    conn.row_factory = sqlite3.Row
    return conn

# Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/monitoring')
def monitoring():
    """Live monitoring page"""
    return render_template('monitoring.html')

@app.route('/configuration')
def configuration():
    """Configuration management"""
    # Load current configuration
    try:
        with open('camera_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    # Load users
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users WHERE active = 1').fetchall()
    conn.close()
    
    return render_template('configuration.html', config=config, users=users)

@app.route('/users')
def users():
    """User management"""
    conn = get_db_connection()
    users_list = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return render_template('users.html', users=users_list)

@app.route('/logs')
def logs():
    """Detection logs and history"""
    conn = get_db_connection()
    detections = conn.execute('''
        SELECT * FROM detection_logs 
        ORDER BY timestamp DESC 
        LIMIT 100
    ''').fetchall()
    conn.close()
    return render_template('logs.html', detections=detections)

@app.route('/settings')
def settings():
    """System settings"""
    return render_template('settings.html')

# API Routes
@app.route('/api/start_monitoring', methods=['POST'])
def api_start_monitoring():
    """Start monitoring via API"""
    global camera_agent
    
    if not camera_agent:
        camera_agent = WebCameraAgent()
        if not camera_agent.initialize():
            error_message = camera_agent.last_error or 'Failed to initialize camera'
            camera_agent = None
            return jsonify({'success': False, 'error': error_message})
    
    if camera_agent.start_monitoring():
        return jsonify({'success': True, 'message': 'Monitoring started'})
    else:
        error_message = camera_agent.last_error or 'Monitoring already active'
        return jsonify({'success': False, 'error': error_message})

@app.route('/api/stop_monitoring', methods=['POST'])
def api_stop_monitoring():
    """Stop monitoring via API"""
    global camera_agent
    
    if camera_agent:
        camera_agent.stop_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring stopped'})
    else:
        return jsonify({'success': False, 'error': 'No active monitoring'})

@app.route('/api/status')
def api_status():
    """Get system status"""
    global camera_agent
    
    status = {
        'monitoring_active': camera_agent.is_monitoring if camera_agent else False,
        'camera_available': check_camera_availability(),
        'ollama_available': check_ollama_availability(),
        'stats': camera_agent._serialize_stats() if camera_agent else {}
    }
    
    return jsonify(status)

@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
    """User management API"""
    conn = get_db_connection()
    
    if request.method == 'POST':
        data = request.json
        try:
            conn.execute('''
                INSERT INTO users (email, name, role)
                VALUES (?, ?, ?)
            ''', (data['email'], data['name'], data.get('role', 'user')))
            conn.commit()
            
            # Update configuration with new user
            update_email_recipients()
            
            return jsonify({'success': True, 'message': 'User added successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
        finally:
            conn.close()
    
    else:
        users = conn.execute('SELECT * FROM users WHERE active = 1').fetchall()
        conn.close()
        return jsonify([dict(user) for user in users])

@app.route('/api/users/<int:user_id>', methods=['DELETE', 'PUT'])
def api_delete_user(user_id):
    """Delete or update user"""
    if request.method == 'DELETE':
        conn = get_db_connection()
        conn.execute('UPDATE users SET active = 0 WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        update_email_recipients()
        return jsonify({'success': True, 'message': 'User deactivated'})
    
    elif request.method == 'PUT':
        data = request.json
        conn = get_db_connection()
        try:
            conn.execute('''
                UPDATE users 
                SET email = ?, name = ?, role = ?
                WHERE id = ?
            ''', (data['email'], data['name'], data.get('role', 'user'), user_id))
            conn.commit()
            
            # Update configuration with updated user email
            update_email_recipients()
            
            return jsonify({'success': True, 'message': 'User updated successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
        finally:
            conn.close()

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Configuration management API"""
    if request.method == 'POST':
        data = request.json
        try:
            # Save configuration
            with open('camera_config.yaml', 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            # Log configuration change
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO config_history (config_type, changes)
                VALUES (?, ?)
            ''', ('camera_config', json.dumps(data)))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': 'Configuration updated'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    else:
        try:
            with open('camera_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return jsonify(config)
        except:
            return jsonify({})

@app.route('/api/test_camera')
def api_test_camera():
    """Test camera functionality"""
    global camera_agent
    try:
        # Try to open camera temporarily for testing
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return jsonify({'success': True, 'message': 'Camera test successful'})
            else:
                return jsonify({'success': False, 'error': 'Failed to capture frame'})
        else:
            return jsonify({'success': False, 'error': 'Cannot open camera'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_ollama')
def api_test_ollama():
    """Test Ollama connection"""
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'Ollama connection successful'})
        else:
            return jsonify({'success': False, 'error': f'Ollama returned status {response.status_code}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recent_images')
def api_recent_images():
    """Get recent detection images"""
    try:
        detected_dir = Path('detected_images')
        processed_dir = Path('processed_frames')
        
        detected_images = []
        processed_images = []
        
        if detected_dir.exists():
            for img_file in sorted(detected_dir.glob('*.jpg'), key=os.path.getmtime, reverse=True)[:10]:
                with open(img_file, 'rb') as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                detected_images.append({
                    'filename': img_file.name,
                    'timestamp': datetime.fromtimestamp(img_file.stat().st_mtime).isoformat(),
                    'image_b64': img_b64
                })
        
        if processed_dir.exists():
            for img_file in sorted(processed_dir.glob('*.jpg'), key=os.path.getmtime, reverse=True)[:5]:
                with open(img_file, 'rb') as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                processed_images.append({
                    'filename': img_file.name,
                    'timestamp': datetime.fromtimestamp(img_file.stat().st_mtime).isoformat(),
                    'image_b64': img_b64
                })
        
        return jsonify({
            'detected': detected_images,
            'processed': processed_images
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/logs', methods=['GET', 'DELETE'])
def api_logs():
    """Get or clear detection logs"""
    if request.method == 'GET':
        try:
            conn = get_db_connection()
            logs = conn.execute('''
                SELECT id, timestamp, confidence, response, image_path, frame_number, reason
                FROM detection_logs 
                ORDER BY timestamp DESC
            ''').fetchall()
            conn.close()
            
            logs_list = []
            for log in logs:
                logs_list.append({
                    'id': log['id'],
                    'timestamp': log['timestamp'],
                    'confidence': log['confidence'],
                    'response': log['response'],
                    'image_path': log['image_path'],
                    'frame_number': log['frame_number'],
                    'reason': log['reason'],
                    'detected': log['confidence'] is not None and log['confidence'] > 0
                })
            
            return jsonify({'success': True, 'logs': logs_list})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            conn = get_db_connection()
            conn.execute('DELETE FROM detection_logs')
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'All logs cleared'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/api/logs/<int:log_id>', methods=['DELETE'])
def api_delete_log(log_id):
    """Delete a specific log entry"""
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM detection_logs WHERE id = ?', (log_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Log deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/logs/export', methods=['POST'])
def api_export_logs():
    """Export logs as CSV"""
    try:
        data = request.get_json()
        format_type = data.get('format', 'csv')
        
        conn = get_db_connection()
        logs = conn.execute('''
            SELECT timestamp, confidence, response, image_path, frame_number, reason
            FROM detection_logs 
            ORDER BY timestamp DESC
        ''').fetchall()
        conn.close()
        
        if format_type == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Timestamp', 'Confidence', 'Response', 'Image Path', 'Frame Number', 'Reason'])
            
            for log in logs:
                writer.writerow([
                    log['timestamp'],
                    log['confidence'],
                    log['response'],
                    log['image_path'],
                    log['frame_number'],
                    log['reason']
                ])
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=detection_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            )
        else:
            logs_list = [dict(log) for log in logs]
            return Response(
                json.dumps(logs_list, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename=detection_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'}
            )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/<path:image_path>')
def api_serve_image(image_path):
    """Serve detection images"""
    try:
        # Sanitize and resolve the image path
        safe_path = Path(image_path).resolve()
        
        # Check if path is within allowed directories
        detected_dir = DETECTED_IMAGES_DIR.resolve()
        processed_dir = (DATA_DIR / 'processed_frames').resolve()
        
        if not (str(safe_path).startswith(str(detected_dir)) or str(safe_path).startswith(str(processed_dir))):
            return jsonify({'error': 'Access denied'}), 403
        
        if safe_path.exists() and safe_path.is_file():
            return send_file(safe_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status')
def api_system_status():
    """Get system status information"""
    try:
        import psutil
        
        status = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'camera_available': check_camera_availability(),
            'ollama_available': check_ollama_availability()
        }
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/system/environment', methods=['GET', 'POST'])
def api_system_environment():
    """Get or update system environment variables"""
    if request.method == 'GET':
        try:
            env_vars = {
                'CAMERA_INDEX': os.getenv('CAMERA_INDEX', '0'),
                'OLLAMA_URL': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
                'VISION_MODEL': os.getenv('VISION_MODEL', 'gemma3:4b'),
                'DIFF_THRESHOLD': os.getenv('DIFF_THRESHOLD', '0.80'),
                'CAPTURE_INTERVAL': os.getenv('CAPTURE_INTERVAL', '5'),
                'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
            }
            return jsonify({'success': True, 'environment': env_vars})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            # Note: This only updates in-memory; for persistence, update .env file
            for key, value in data.items():
                os.environ[key] = str(value)
            return jsonify({'success': True, 'message': 'Environment variables updated (in-memory only)'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/api/video_feed')
def video_feed():
    """Stream live video frames from camera"""
    def generate():
        try:
            while True:
                if camera_agent and camera_agent.last_frame:
                    # Use the latest frame captured by monitoring loop
                    frame_data = base64.b64decode(camera_agent.last_frame['image_b64'])
                    
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Video feed error: {e}")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture_frame')
def capture_frame():
    """Get the latest captured frame"""
    try:
        if camera_agent and camera_agent.last_frame:
            return jsonify({
                'success': True,
                'image_b64': camera_agent.last_frame['image_b64'],
                'timestamp': camera_agent.last_frame['timestamp'],
                'metadata': {
                    'frame_number': camera_agent.last_frame.get('frame_number', 0),
                    'source': 'live_stream'
                }
            })
        
        return jsonify({'success': False, 'error': 'No frames available yet'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Helper functions
def check_camera_availability():
    """Check if camera is available"""
    global camera_agent
    if camera_agent and camera_agent.camera:
        try:
            return camera_agent.camera.isOpened()
        except:
            return False
    return False

def check_ollama_availability():
    """Check if Ollama is available"""
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        return response.status_code == 200
    except:
        return False

def update_email_recipients():
    """Update email recipients in configuration based on active users"""
    try:
        # Get active users with valid emails
        conn = get_db_connection()
        users = conn.execute('SELECT email FROM users WHERE active = 1 AND email IS NOT NULL AND email != ""').fetchall()
        conn.close()
        
        emails = [user['email'] for user in users]
        
        if not emails:
            logger.warning("No active users found to update email recipients")
            return
        
        # Update configuration
        config_path = 'camera_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            logger.error("Failed to load camera configuration")
            return
        
        rules = config.get('rules', [])
        if not rules:
            logger.warning("No rules found in configuration to update")
            return
        
        updated = False
        for rule in rules:
            # Check for actions.email.to structure
            if 'actions' in rule and isinstance(rule['actions'], dict):
                if 'email' in rule['actions'] and isinstance(rule['actions']['email'], dict):
                    rule['actions']['email']['to'] = emails
                    updated = True
                    logger.info(f"Updated rule '{rule.get('id', 'unknown')}' with {len(emails)} recipients")
            
            # Check for legacy email.to structure
            elif 'email' in rule and isinstance(rule['email'], dict):
                rule['email']['to'] = emails
                updated = True
                logger.info(f"Updated legacy rule '{rule.get('id', 'unknown')}' with {len(emails)} recipients")
        
        if updated:
            # Write back to file
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            logger.info(f"Email recipients updated successfully. Recipients: {', '.join(emails)}")
        else:
            logger.warning("No rules were updated with email recipients")
            
    except Exception as e:
        logger.error(f"Failed to update email recipients: {e}", exc_info=True)

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to ZEDEDA Camera Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    # Initialize database schema before serving requests
    init_db()
    update_email_recipients()
    
    # Run the app
    print("🌐 Starting ZEDEDA Camera Agent Web Interface...")
    print("📱 Access the interface at: http://localhost:8080")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)