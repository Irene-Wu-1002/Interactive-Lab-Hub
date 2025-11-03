"""
Multi-Person Smile-Check Camera
A simple interaction prototype that detects multiple faces and checks if everyone is smiling.
Uses MediaPipe Face Detection (BlazeFace) for face detection and image analysis for smile estimation.

Interaction Concept:
- Detects multiple faces simultaneously
- Estimates smile score for each face using mouth region analysis
- Triggers feedback when not everyone is smiling (smile score < 0.65 for >1 second)
- Multiple feedback modes: audio, LED simulation, screen overlay, progress bar
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

class SmileCheckCamera:
    def __init__(self):
        # Initialize MediaPipe Face Detection (BlazeFace)
        # Using face_detection instead of face_mesh to avoid UnicodeDecodeError
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (0-5 meters), 1 for full-range
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Face tracking: store (face_id, smile_score, last_update_time, below_threshold_since)
        # below_threshold_since: timestamp when face first dropped below threshold (None if above threshold)
        self.face_tracking = {}
        self.face_id_counter = 0
        
        # Feedback state
        self.feedback_active = False
        self.feedback_start_time = None
        self.led_state = False
        self.last_led_toggle = time.time()
        self.led_blink_rate = 0.5  # seconds
        
        # Smile detection parameters
        self.smile_threshold = 0.65
        self.feedback_delay = 1.0  # seconds before triggering feedback
        
        # Audio feedback (using system beep or pygame if available)
        try:
            import pygame
            pygame.mixer.init()
            # Generate a simple beep sound
            self.use_pygame = True
        except ImportError:
            self.use_pygame = False
            print("PyGame not available. Audio feedback disabled.")
            print("Install with: pip install pygame")
    
    def calculate_smile_score(self, face_bbox, image):
        """
        Calculate smile score based on mouth region analysis within detected face.
        Uses edge detection and contour analysis on the mouth region.
        Higher score indicates a smile.
        """
        h, w = image.shape[:2]
        
        # Extract face bounding box coordinates
        # MediaPipe returns normalized coordinates (xmin, ymin, width, height)
        x_min = int(face_bbox.xmin * w)
        y_min = int(face_bbox.ymin * h)
        x_max = int((face_bbox.xmin + face_bbox.width) * w)
        y_max = int((face_bbox.ymin + face_bbox.height) * h)
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.5  # Default neutral
        
        # Extract face region
        face_region = image[y_min:y_max, x_min:x_max]
        
        if face_region.size == 0:
            return 0.5
        
        # Estimate mouth region (typically in lower 1/3 of face, centered horizontally)
        face_height = y_max - y_min
        face_width = x_max - x_min
        
        # Mouth region is approximately in the lower third of the face
        mouth_y_start = int(face_height * 0.55)
        mouth_y_end = int(face_height * 0.85)
        mouth_x_start = int(face_width * 0.25)
        mouth_x_end = int(face_width * 0.75)
        
        if mouth_y_end <= mouth_y_start or mouth_x_end <= mouth_x_start:
            return 0.5
        
        mouth_region = face_region[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        
        if mouth_region.size == 0:
            return 0.5
        
        # Convert to grayscale if needed
        if len(mouth_region.shape) == 3:
            mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            mouth_gray = mouth_region
        
        # Apply edge detection
        edges = cv2.Canny(mouth_gray, 50, 150)
        
        # Calculate horizontal edge density (smiles create more horizontal edges)
        # Count horizontal edges (smiles curve upward)
        horizontal_kernel = np.array([[-1, -1, -1],
                                      [ 2,  2,  2],
                                      [-1, -1, -1]])
        horizontal_edges = cv2.filter2D(edges, -1, horizontal_kernel)
        
        # Calculate metrics
        total_edges = np.sum(edges > 0)
        horizontal_edge_strength = np.sum(horizontal_edges > 0)
        
        # Smile heuristic: higher horizontal edge ratio + mouth width
        mouth_width = mouth_x_end - mouth_x_start
        mouth_height = mouth_y_end - mouth_y_start
        
        if total_edges > 0 and mouth_height > 0:
            edge_ratio = horizontal_edge_strength / total_edges if total_edges > 0 else 0
            width_ratio = mouth_width / mouth_height
            
            # Combine metrics: wider mouth and upward curve suggests smile
            smile_score = (edge_ratio * 0.6 + min(width_ratio / 3.0, 1.0) * 0.4)
            smile_score = np.clip(smile_score, 0.0, 1.0)
        else:
            smile_score = 0.5  # Default neutral
        
        return smile_score
    
    def play_audio_feedback(self):
        """Play audio feedback (ding tone)"""
        if self.use_pygame:
            try:
                import pygame
                # Generate a simple beep (440 Hz for 0.2 seconds)
                sample_rate = 44100
                duration = 0.2
                frequency = 440
                frames = int(duration * sample_rate)
                arr = np.zeros((frames, 2), dtype=np.float32)
                for i in range(frames):
                    arr[i][0] = np.sin(2 * np.pi * frequency * i / sample_rate)
                    arr[i][1] = arr[i][0]
                sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
                sound.play()
            except Exception as e:
                print(f"Audio error: {e}")
        else:
            # Fallback: print to console
            print("\a", end='', flush=True)  # System beep
    
    def update_face_tracking(self, detected_faces, image, current_time):
        """Update tracking for detected faces and calculate smile scores"""
        # Reset all faces as not seen in this frame
        faces_seen = set()
        
        for detection in detected_faces:
            # Get face bounding box from MediaPipe Face Detection
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate face center for tracking
            face_center_x = bbox.xmin + bbox.width / 2
            face_center_y = bbox.ymin + bbox.height / 2
            
            # Find closest existing face or create new one
            best_match_id = None
            min_distance = float('inf')
            
            for face_id, (x, y, score, last_time, below_threshold_since) in self.face_tracking.items():
                distance = math.hypot(x - face_center_x, y - face_center_y)
                if distance < min_distance and distance < 0.1:  # Threshold for matching
                    min_distance = distance
                    best_match_id = face_id
            
            if best_match_id is None:
                # New face detected
                best_match_id = self.face_id_counter
                self.face_id_counter += 1
                below_threshold_since = None
            else:
                # Get existing below_threshold_since to preserve timing
                _, _, old_score, _, below_threshold_since = self.face_tracking[best_match_id]
            
            # Calculate smile score for this face
            smile_score = self.calculate_smile_score(bbox, image)
            
            # Update below_threshold_since tracking
            if smile_score < self.smile_threshold:
                if below_threshold_since is None:
                    # First time dropping below threshold
                    below_threshold_since = current_time
            else:
                # Above threshold, reset
                below_threshold_since = None
            
            faces_seen.add(best_match_id)
            self.face_tracking[best_match_id] = (face_center_x, face_center_y, smile_score, current_time, below_threshold_since)
        
        # Remove faces not seen in this frame
        faces_to_remove = [fid for fid in self.face_tracking.keys() if fid not in faces_seen]
        for fid in faces_to_remove:
            del self.face_tracking[fid]
    
    def check_smile_status(self):
        """Check if all faces are smiling and trigger feedback if needed"""
        current_time = time.time()
        all_smiling = True
        min_smile_score = 1.0
        
        # If no faces detected, consider as "all smiling" (no feedback needed)
        if len(self.face_tracking) == 0:
            self.feedback_active = False
            self.feedback_start_time = None
            return True, 1.0
        
        for face_id, (x, y, score, last_time, below_threshold_since) in self.face_tracking.items():
            if score is None:
                all_smiling = False
                min_smile_score = 0.0
                break
            
            if score < self.smile_threshold:
                all_smiling = False
                min_smile_score = min(min_smile_score, score)
                
                # Check if this face has been below threshold for more than delay time
                if below_threshold_since is not None:
                    time_below_threshold = current_time - below_threshold_since
                    if time_below_threshold > self.feedback_delay:
                        if not self.feedback_active:
                            self.feedback_active = True
                            self.feedback_start_time = current_time
                        return False, min_smile_score
        
        # All faces are smiling
        self.feedback_active = False
        self.feedback_start_time = None
        return True, min_smile_score
    
    def draw_feedback(self, image):
        """Draw various feedback modes on the image"""
        h, w = image.shape[:2]
        
        # Visual smile sync bar
        bar_width = int(w * 0.6)
        bar_height = 30
        bar_x = (w - bar_width) // 2
        bar_y = 30
        
        # Calculate overall smile status
        if len(self.face_tracking) > 0:
            scores = [score for _, _, score, _, _ in self.face_tracking.values() if score is not None]
            if scores:
                avg_score = np.mean(scores)
                min_score = min(scores)
            else:
                avg_score = 0.5
                min_score = 0.5
        else:
            avg_score = 0.5
            min_score = 0.5
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress bar (green when smiling, red when not)
        progress = int(bar_width * min_score)
        color = (0, 255, 0) if min_score >= self.smile_threshold else (0, 0, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), color, -1)
        # Border
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        # Text
        cv2.putText(image, f'Smile Status: {int(min_score*100)}%', 
                   (bar_x + 10, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Red border when feedback is active
        if self.feedback_active:
            border_thickness = 10
            cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), border_thickness)
            
            # Warning text
            text = "Someone isn't smiling yet!"
            font_scale = 1.2
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)
            text_x = (w - text_width) // 2
            text_y = h - 50
            # Background for text
            cv2.rectangle(image, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + baseline + 10),
                         (0, 0, 255), -1)
            cv2.putText(image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness)
        
        # LED simulation (flashing when feedback active)
        if self.feedback_active:
            current_time = time.time()
            if current_time - self.last_led_toggle > self.led_blink_rate:
                self.led_state = not self.led_state
                self.last_led_toggle = current_time
            
            # Draw LED indicator (top right)
            led_size = 30
            led_x = w - 50
            led_y = 50
            led_color = (0, 255, 255) if self.led_state else (0, 150, 150)  # Cyan LED
            cv2.circle(image, (led_x, led_y), led_size, led_color, -1)
            cv2.circle(image, (led_x, led_y), led_size, (255, 255, 255), 2)
            cv2.putText(image, "LED", (led_x - 20, led_y + led_size + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, image):
        """Process a single frame and return annotated image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        current_time = time.time()
        
        h, w = image.shape[:2]
        
        # Track faces and calculate smile scores
        if results.detections:
            # Update face tracking and calculate smile scores in one pass
            self.update_face_tracking(results.detections, image, current_time)
            
            # Draw bounding boxes and smile scores for each detected face
            for idx, detection in enumerate(results.detections):
                # Get face bounding box
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                x_max = int((bbox.xmin + bbox.width) * w)
                y_max = int((bbox.ymin + bbox.height) * h)
                
                # Get face center for matching display
                face_center_x = bbox.xmin + bbox.width / 2
                face_center_y = bbox.ymin + bbox.height / 2
                
                # Find corresponding face ID in tracking
                best_match_id = None
                min_distance = float('inf')
                for face_id, (x, y, score, last_time, below_threshold_since) in self.face_tracking.items():
                    distance = math.hypot(x - face_center_x, y - face_center_y)
                    if distance < min_distance and distance < 0.1:
                        min_distance = distance
                        best_match_id = face_id
                
                # Get smile score for this face
                if best_match_id is not None:
                    _, _, smile_score, _, _ = self.face_tracking[best_match_id]
                else:
                    smile_score = self.calculate_smile_score(bbox, image)
                
                # Draw bounding box
                box_color = (0, 255, 0) if smile_score >= self.smile_threshold else (0, 0, 255)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)
                
                # Display smile score
                face_label = best_match_id if best_match_id is not None else idx
                score_text = f"Face {face_label}: {smile_score:.2f}"
                cv2.putText(image, score_text, (x_min, max(y_min - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Check smile status and trigger feedback
        all_smiling, min_score = self.check_smile_status()
        
        # Trigger audio feedback (once when feedback activates)
        if self.feedback_active and self.feedback_start_time:
            if time.time() - self.feedback_start_time < 0.3:  # Play once at start
                self.play_audio_feedback()
        
        # Draw feedback overlays
        self.draw_feedback(image)
        
        # Display status
        h, w = image.shape[:2]
        status_text = f"Faces: {len(self.face_tracking)} | All Smiling: {all_smiling}"
        cv2.putText(image, status_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

def main():
    """Main function to run the smile check camera"""
    print("Starting Multi-Person Smile-Check Camera...")
    print("Press 'q' to quit")
    print(f"Feedback threshold: smile score < 0.65 for >1 second")
    
    camera = SmileCheckCamera()
    
    ################################
    wCam, hCam = 640, 480
    ################################
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    pTime = 0
    
    while True:
        success, img = cap.read()
        
        if not success:
            print("Error: Could not read frame")
            break
        
        # Process frame - this adds annotations
        img = camera.process_frame(img)
        
        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        
        # Display frame - always show, even if no faces detected
        cv2.imshow("Img", img)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

