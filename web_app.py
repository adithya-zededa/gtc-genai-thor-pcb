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
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
import threading
import time
from camera_agent import MonitorDetectionAgent
import cv2
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
camera_agent = None
monitoring_active = False
monitoring_thread = None

class WebCameraAgent:
    """Web-integrated camera agent with real-time updates"""
    
    def __init__(self):
        self.agent = None
        self.is_monitoring = False
        self.last_frame = None
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'uptime_start': datetime.now()
        }
    
    def initialize(self):
        """Initialize the camera agent"""
        try:
            self.agent = MonitorDetectionAgent()
            return self.agent.initialize()
        except Exception as e:
            print(f"Failed to initialize camera agent: {e}")
            return False
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.is_monitoring:
            return False
        
        self.is_monitoring = True
        thread = threading.Thread(target=self._monitoring_loop)
        thread.daemon = True
        thread.start()
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
    
    def _monitoring_loop(self):
        """Main monitoring loop with web integration"""
        while self.is_monitoring:
            try:
                # Capture frame
                capture_result = self.agent.camera_monitor.capture_frame_smart()
                if capture_result:
                    image_data, frame_metadata = capture_result
                    self.stats['processed_frames'] += 1
                    
                    # Store latest frame for web display
                    self.last_frame = {
                        'image_b64': base64.b64encode(image_data).decode('utf-8'),
                        'metadata': frame_metadata,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Analyze frame
                    event = self.agent.analyze_frame(image_data, frame_metadata)
                    if event:
                        self.stats['detections'] += 1
                        
                        # Process alerts
                        if self.agent.process_detection(event):
                            self.stats['alerts_sent'] += 1
                        
                        # Emit real-time update
                        socketio.emit('detection_event', {
                            'event': {
                                'timestamp': event.timestamp,
                                'confidence': event.confidence,
                                'response': event.full_response[:200] + '...' if len(event.full_response) > 200 else event.full_response
                            },
                            'stats': self.stats.copy()
                        })
                    
                    # Emit frame update
                    socketio.emit('frame_update', {
                        'frame': self.last_frame,
                        'stats': self.stats.copy()
                    })
                
                self.stats['total_frames'] += 1
                time.sleep(1)  # Reduced for more responsive web updates
                
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                time.sleep(5)

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
            return jsonify({'success': False, 'error': 'Failed to initialize camera'})
    
    if camera_agent.start_monitoring():
        return jsonify({'success': True, 'message': 'Monitoring started'})
    else:
        return jsonify({'success': False, 'error': 'Monitoring already active'})

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
        'stats': camera_agent.stats if camera_agent else {}
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

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    """Delete user"""
    conn = get_db_connection()
    conn.execute('UPDATE users SET active = 0 WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    update_email_recipients()
    return jsonify({'success': True, 'message': 'User deactivated'})

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
    try:
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

# Helper functions
def check_camera_availability():
    """Check if camera is available"""
    try:
        cap = cv2.VideoCapture(0)
        available = cap.isOpened()
        cap.release()
        return available
    except:
        return False

def check_ollama_availability():
    """Check if Ollama is available"""
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        return response.status_code == 200
    except:
        return False

def update_email_recipients():
    """Update email recipients in configuration based on users"""
    try:
        # Get active users
        conn = get_db_connection()
        users = conn.execute('SELECT email FROM users WHERE active = 1').fetchall()
        conn.close()
        
        emails = [user['email'] for user in users]
        
        # Update configuration
        with open('camera_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'rules' in config and len(config['rules']) > 0:
            config['rules'][0]['email']['to'] = emails
            
            with open('camera_config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Failed to update email recipients: {e}")

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
    # Initialize database
    init_db()
    
    # Run the app
    print("🌐 Starting ZEDEDA Camera Agent Web Interface...")
    print("📱 Access the interface at: http://localhost:8080")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)