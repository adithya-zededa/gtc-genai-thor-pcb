#!/usr/bin/env python3
"""
Camera Feed Publisher
Implements a publisher-subscriber pattern for camera frame distribution.
The publisher captures frames from the camera and distributes them to multiple subscribers.
"""

import cv2
import base64
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Full
from typing import Dict, Optional, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    """Represents a single camera frame with metadata."""
    frame_number: int
    timestamp: str
    image_data: bytes  # JPEG encoded
    image_b64: str  # Base64 encoded for web transmission
    raw_frame: Any  # OpenCV frame (numpy array)
    width: int
    height: int


class CameraFeedPublisher:
    """
    Publisher that captures frames from camera and distributes to subscribers.
    Uses a thread-safe queue mechanism for each subscriber.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the camera feed publisher.
        
        Args:
            camera_index: Camera device index (default 0 for /dev/video0)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 0.033
        
        # Camera state
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Frame tracking
        self.frame_number = 0
        self.last_frame: Optional[CameraFrame] = None
        
        # Subscriber management
        self._subscribers: Dict[str, Queue] = {}
        self._subscriber_callbacks: Dict[str, Optional[Callable]] = {}
        self._max_queue_size = 10  # Max frames queued per subscriber
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'subscribers_count': 0,
            'last_error': None
        }
        
        logger.info(f"CameraFeedPublisher initialized: camera={camera_index}, resolution={width}x{height}, fps={fps}")
    
    def initialize_camera(self) -> bool:
        """Initialize the camera device."""
        with self._lock:
            if self.camera and self.camera.isOpened():
                logger.debug("Camera already initialized")
                return True
            
            try:
                self.camera = cv2.VideoCapture(self.camera_index)
                if not self.camera.isOpened():
                    error_msg = f"Failed to open camera at index {self.camera_index}"
                    logger.error(error_msg)
                    self.stats['last_error'] = error_msg
                    return False
                
                # Configure camera
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Verify settings
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Camera initialized: {actual_width}x{actual_height}")
                
                return True
                
            except Exception as e:
                error_msg = f"Camera initialization error: {e}"
                logger.error(error_msg, exc_info=True)
                self.stats['last_error'] = error_msg
                return False
    
    def start(self) -> bool:
        """Start the camera feed publisher."""
        with self._lock:
            if self.is_running:
                logger.warning("Publisher already running")
                return True
            
            if not self.initialize_camera():
                return False
            
            self.is_running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraPublisher")
            self._capture_thread.start()
            
            logger.info("Camera feed publisher started")
            return True
    
    def stop(self) -> None:
        """Stop the camera feed publisher and release resources."""
        with self._lock:
            if not self.is_running:
                return
            
            logger.info("Stopping camera feed publisher...")
            self.is_running = False
            
            # Wait for capture thread to finish
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            
            # Release camera
            if self.camera:
                self.camera.release()
                self.camera = None
            
            # Clear all subscriber queues
            for queue in self._subscribers.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except:
                        pass
            
            logger.info("Camera feed publisher stopped")
    
    def subscribe(self, subscriber_id: str, callback: Optional[Callable[[CameraFrame], None]] = None) -> bool:
        """
        Subscribe to the camera feed.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            callback: Optional callback function to be called for each frame
        
        Returns:
            True if subscription successful, False otherwise
        """
        with self._lock:
            if subscriber_id in self._subscribers:
                logger.warning(f"Subscriber '{subscriber_id}' already exists")
                return False
            
            self._subscribers[subscriber_id] = Queue(maxsize=self._max_queue_size)
            self._subscriber_callbacks[subscriber_id] = callback
            self.stats['subscribers_count'] = len(self._subscribers)
            
            logger.info(f"Subscriber '{subscriber_id}' added. Total subscribers: {self.stats['subscribers_count']}")
            return True
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from the camera feed.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
        
        Returns:
            True if unsubscription successful, False otherwise
        """
        with self._lock:
            if subscriber_id not in self._subscribers:
                logger.warning(f"Subscriber '{subscriber_id}' not found")
                return False
            
            # Clear the queue
            queue = self._subscribers[subscriber_id]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass
            
            del self._subscribers[subscriber_id]
            del self._subscriber_callbacks[subscriber_id]
            self.stats['subscribers_count'] = len(self._subscribers)
            
            logger.info(f"Subscriber '{subscriber_id}' removed. Total subscribers: {self.stats['subscribers_count']}")
            return True
    
    def get_frame(self, subscriber_id: str, timeout: float = 1.0) -> Optional[CameraFrame]:
        """
        Get the next frame for a subscriber (blocking).
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            timeout: Maximum time to wait for a frame (seconds)
        
        Returns:
            CameraFrame if available, None otherwise
        """
        if subscriber_id not in self._subscribers:
            logger.warning(f"Subscriber '{subscriber_id}' not found")
            return None
        
        try:
            queue = self._subscribers[subscriber_id]
            return queue.get(timeout=timeout)
        except:
            return None
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """
        Get the most recent captured frame without subscribing.
        Useful for one-time frame requests.
        
        Returns:
            Latest CameraFrame if available, None otherwise
        """
        with self._lock:
            return self.last_frame
    
    def _capture_loop(self) -> None:
        """Main capture loop that runs in a separate thread."""
        logger.info("Camera capture loop started")
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                # Capture frame
                if not self.camera or not self.camera.isOpened():
                    logger.error("Camera not available in capture loop")
                    time.sleep(1.0)
                    continue
                
                ret, raw_frame = self.camera.read()
                if not ret or raw_frame is None:
                    logger.warning("Failed to read frame from camera")
                    self.stats['frames_dropped'] += 1
                    time.sleep(0.1)
                    continue
                
                # Encode frame to JPEG
                encode_success, buffer = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not encode_success:
                    logger.warning("Failed to encode frame to JPEG")
                    self.stats['frames_dropped'] += 1
                    continue
                
                image_data = buffer.tobytes()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                # Create frame object
                self.frame_number += 1
                frame = CameraFrame(
                    frame_number=self.frame_number,
                    timestamp=datetime.now().isoformat(),
                    image_data=image_data,
                    image_b64=image_b64,
                    raw_frame=raw_frame,
                    width=raw_frame.shape[1],
                    height=raw_frame.shape[0]
                )
                
                # Update statistics
                self.stats['frames_captured'] += 1
                
                # Store latest frame
                with self._lock:
                    self.last_frame = frame
                
                # Publish to all subscribers
                self._publish_frame(frame)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}", exc_info=True)
                self.stats['last_error'] = str(e)
                time.sleep(0.1)
            
            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Camera capture loop ended")
    
    def _publish_frame(self, frame: CameraFrame) -> None:
        """Publish a frame to all subscribers."""
        subscribers_to_remove = []
        
        with self._lock:
            for subscriber_id, queue in self._subscribers.items():
                try:
                    # Non-blocking put - drop frame if queue is full
                    queue.put_nowait(frame)
                    
                    # Call callback if provided
                    callback = self._subscriber_callbacks.get(subscriber_id)
                    if callback:
                        try:
                            callback(frame)
                        except Exception as e:
                            logger.error(f"Error in callback for subscriber '{subscriber_id}': {e}")
                    
                except Full:
                    # Queue is full, drop the oldest frame and add new one
                    try:
                        queue.get_nowait()  # Remove oldest
                        queue.put_nowait(frame)  # Add newest
                    except:
                        pass
                except Exception as e:
                    logger.error(f"Error publishing to subscriber '{subscriber_id}': {e}")
                    subscribers_to_remove.append(subscriber_id)
            
            # Clean up failed subscribers
            for subscriber_id in subscribers_to_remove:
                logger.warning(f"Removing failed subscriber: {subscriber_id}")
                self._subscribers.pop(subscriber_id, None)
                self._subscriber_callbacks.pop(subscriber_id, None)
            
            if subscribers_to_remove:
                self.stats['subscribers_count'] = len(self._subscribers)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        with self._lock:
            return {
                'is_running': self.is_running,
                'frame_number': self.frame_number,
                'frames_captured': self.stats['frames_captured'],
                'frames_dropped': self.stats['frames_dropped'],
                'subscribers_count': self.stats['subscribers_count'],
                'last_error': self.stats['last_error'],
                'camera_available': self.camera.isOpened() if self.camera else False
            }


# Global singleton instance
_publisher_instance: Optional[CameraFeedPublisher] = None
_publisher_lock = threading.Lock()


def get_camera_publisher(camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> CameraFeedPublisher:
    """
    Get or create the global camera feed publisher instance.
    
    Args:
        camera_index: Camera device index
        width: Frame width
        height: Frame height
        fps: Target frames per second
    
    Returns:
        CameraFeedPublisher instance
    """
    global _publisher_instance
    
    with _publisher_lock:
        if _publisher_instance is None:
            _publisher_instance = CameraFeedPublisher(camera_index, width, height, fps)
        return _publisher_instance


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Camera Feed Publisher Demo")
    print("=" * 50)
    
    publisher = CameraFeedPublisher(camera_index=0, width=640, height=480, fps=10)
    
    if not publisher.start():
        print("Failed to start publisher")
        exit(1)
    
    # Subscribe a test subscriber
    def frame_callback(frame: CameraFrame):
        print(f"Frame {frame.frame_number} received at {frame.timestamp}")
    
    publisher.subscribe("test_subscriber", callback=frame_callback)
    
    try:
        print("Publishing frames... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
            stats = publisher.get_stats()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        publisher.unsubscribe("test_subscriber")
        publisher.stop()
        print("Demo complete")
