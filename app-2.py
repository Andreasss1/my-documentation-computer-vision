from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class YOLODetectionSystem:
    def __init__(self, model_path="oppo.pt"):
        self.model = YOLO(model_path)
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.detection_results = {
            'has_sg': False,
            'detections': [],
            'fps': 0,
            'stats': {'total': 0, 'pass': 0, 'ng': 0}
        }
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def process_detections(self, results, frame):
        """Process YOLO detection results"""
        has_sg = False
        detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"Class_{class_id}"
                
                detection_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox.tolist()
                }
                detections.append(detection_info)
                
                if class_name.lower() == 'sg':
                    has_sg = True
                
                x1, y1, x2, y2 = bbox.astype(int)
                if class_name.lower() == 'sg':
                    color = (0, 0, 255)  # Red for SG (NG)
                    status = "SG - DEFECT"
                else:
                    color = (0, 255, 255)  # Yellow for other objects
                    status = class_name.upper()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{status} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return has_sg, detections, frame
    
    def update_statistics(self, has_sg):
        """Update production statistics"""
        current_time = time.time()
        if not hasattr(self, 'last_detection_time'):
            self.last_detection_time = 0
        
        if current_time - self.last_detection_time > 3.0:
            self.detection_results['stats']['total'] += 1
            if has_sg:
                self.detection_results['stats']['ng'] += 1
            else:
                self.detection_results['stats']['pass'] += 1
            self.last_detection_time = current_time
    
    def calculate_fps(self):
        """Calculate FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.detection_results['fps'] = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def generate_frames(self):
        """Generate video frames with detections"""
        while self.is_running and self.camera is not None:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            try:
                results = self.model(frame, conf=0.5)
                has_sg, detections, processed_frame = self.process_detections(results, frame)
                
                self.detection_results['has_sg'] = has_sg
                self.detection_results['detections'] = detections
                self.update_statistics(has_sg)
                self.calculate_fps()
                
                status_color = (0, 0, 255) if has_sg else (0, 255, 0)
                status_text = f"FPS: {self.detection_results['fps']} | Status: {'NG - SG DETECTED' if has_sg else 'PASS - NO DEFECTS'}"
                overlay = processed_frame.copy()
                cv2.rectangle(overlay, (5, 5), (len(status_text) * 12 + 10, 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
                cv2.putText(processed_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                self.current_frame = processed_frame.copy()
                
                # Encode frame to base64 for Socket.IO
                ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit detection results and frame
                    socketio.emit('detection_result', {
                        'status': 'NG' if has_sg else 'PASS',
                        'pass_count': self.detection_results['stats']['pass'],
                        'ng_count': self.detection_results['stats']['ng'],
                        'total_count': self.detection_results['stats']['total'],
                        'ng_rate': round((self.detection_results['stats']['ng'] / 
                                       max(1, self.detection_results['stats']['total']) * 100), 1)
                    })
                    socketio.emit('video_frame', {'frame': frame_bytes})
            
            except Exception as e:
                print(f"Detection error: {e}")
                self.current_frame = frame.copy()
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_detection(self):
        """Start detection system"""
        if self.initialize_camera():
            self.is_running = True
            threading.Thread(target=self.generate_frames, daemon=True).start()
            return True
        return False
    
    def stop_detection(self):
        """Stop detection system"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def restart_detection(self):
        """Restart detection system"""
        self.stop_detection()
        time.sleep(0.5)
        return self.start_detection()

# Initialize detection system
detector = YOLODetectionSystem()

@app.route('/')
def index():
    """Main page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oppo Object Detection Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        .video-container {
            position: relative;
            width: 100%;
            height: 0;
            padding-bottom: 75%;
        }
        .video-feed {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .detection-banner {
            transition: background-color 0.3s ease;
        }
        .pass-bg {
            background-color: #00D100;
        }
        .ng-bg {
            background-color: #FF0000;
        }
        .no-detection-bg {
            background: linear-gradient(to right, #FFFFFF, #CCCCCC);
        }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-6">
        <div id="detectionBanner" class="detection-banner rounded-lg shadow-lg mb-6 py-8 text-center no-detection-bg">
            <h1 id="detectionStatus" class="text-5xl font-bold text-gray-800">NO DETECTION</h1>
        </div>

        <div class="flex flex-col lg:flex-row gap-6">
            <div class="flex-1">
                <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                    <div class="video-container">
                        <img id="videoFeed" class="video-feed" src="" alt="Video Feed">
                    </div>
                </div>
            </div>

            <div class="w-full lg:w-80">
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-6">System Controls</h2>
                    
                    <div class="flex flex-col space-y-4">
                        <button id="startBtn" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg transition">
                            Start Detection
                        </button>
                        <button id="stopBtn" class="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition" disabled>
                            Stop Detection
                        </button>
                        <button id="restartBtn" class="bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition" disabled>
                            Restart System
                        </button>
                    </div>

                    <div class="mt-8">
                        <h2 class="text-2xl font-bold text-gray-800 mb-4">Production Statistics</h2>
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span class="font-medium">Pass Count:</span>
                                <span id="passCount" class="font-bold">0</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="font-medium">NG Count:</span>
                                <span id="ngCount" class="font-bold">0</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="font-medium">Total:</span>
                                <span id="totalCount" class="font-bold">0</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="font-medium">NG Rate:</span>
                                <span id="ngRate" class="font-bold">0%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const videoFeed = document.getElementById('videoFeed');
        const detectionBanner = document.getElementById('detectionBanner');
        const detectionStatus = document.getElementById('detectionStatus');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const restartBtn = document.getElementById('restartBtn');
        const passCount = document.getElementById('passCount');
        const ngCount = document.getElementById('ngCount');
        const totalCount = document.getElementById('totalCount');
        const ngRate = document.getElementById('ngRate');

        function updateButtonStates(isRunning) {
            startBtn.disabled = isRunning;
            stopBtn.disabled = !isRunning;
            restartBtn.disabled = !isRunning;
        }

        startBtn.addEventListener('click', () => {
            socket.emit('start_detection');
        });

        stopBtn.addEventListener('click', () => {
            socket.emit('stop_detection');
        });

        restartBtn.addEventListener('click', () => {
            socket.emit('restart_system');
        });

        socket.on('connect', () => {
            console.log('Connected to server');
        });

        socket.on('system_status', (data) => {
            updateButtonStates(data.is_running);
            if (!data.is_running) {
                videoFeed.src = '';
                detectionBanner.className = 'detection-banner rounded-lg shadow-lg mb-6 py-8 text-center no-detection-bg';
                detectionStatus.textContent = 'NO DETECTION';
                detectionStatus.className = 'text-5xl font-bold text-gray-800';
            }
        });

        socket.on('video_frame', (data) => {
            videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
        });

        socket.on('detection_result', (data) => {
            if (data.status === 'PASS') {
                detectionBanner.className = 'detection-banner rounded-lg shadow-lg mb-6 py-8 text-center pass-bg';
                detectionStatus.textContent = 'PASS';
                detectionStatus.className = 'text-5xl font-bold text-white';
            } else if (data.status === 'NG') {
                detectionBanner.className = 'detection-banner rounded-lg shadow-lg mb-6 py-8 text-center ng-bg';
                detectionStatus.textContent = 'NG';
                detectionStatus.className = 'text-5xl font-bold text-white';
            } else {
                detectionBanner.className = 'detection-banner rounded-lg shadow-lg mb-6 py-8 text-center no-detection-bg';
                detectionStatus.textContent = 'NO DETECTION';
                detectionStatus.className = 'text-5xl font-bold text-gray-800';
            }

            passCount.textContent = data.pass_count;
            ngCount.textContent = data.ng_count;
            totalCount.textContent = data.total_count;
            ngRate.textContent = data.ng_rate + '%';
        });
    </script>
</body>
</html>
    '''

@socketio.on('start_detection')
def handle_start_detection():
    if detector.start_detection():
        emit('system_status', {'is_running': True})
    else:
        emit('system_status', {'is_running': False, 'message': 'Failed to initialize camera'})

@socketio.on('stop_detection')
def handle_stop_detection():
    detector.stop_detection()
    emit('system_status', {'is_running': False})

@socketio.on('restart_system')
def handle_restart_system():
    if detector.restart_detection():
        emit('system_status', {'is_running': True})
    else:
        emit('system_status', {'is_running': False, 'message': 'Failed to restart detection'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)