"""
Multi-Person Smile-Check Camera
A simple interaction prototype that detects multiple faces and checks if everyone is smiling.
Uses OpenCV Haar Cascade for face detection and image analysis for smile estimation.

Interaction Concept:
- Detects multiple faces simultaneously
- Estimates smile score for each face using mouth region analysis
- Triggers feedback when not everyone is smiling (smile score < 0.65 for >1 second)
- Multiple feedback modes: audio, LED simulation, screen overlay, progress bar
"""

import cv2
import numpy as np
import time
import math
import os

class SmileCheckCamera:
    def __init__(self):
        # Initialize OpenCV Haar Cascade Face Detection
        # This is more reliable on Raspberry Pi and doesn't require MediaPipe
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Check if cascade file exists, if not try alternative paths
        if not os.path.exists(cascade_path):
            # Try common alternative paths on Raspberry Pi
            alt_paths = [
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_default.xml'
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    cascade_path = alt_path
                    break
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
            print(f"Loaded face detector from: {cascade_path}")
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            print("Falling back to default OpenCV cascade path...")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Face tracking: store (face_id, smile_score, last_update_time, below_threshold_since, smoothed_score)
        # below_threshold_since: timestamp when face first dropped below threshold (None if above threshold)
        # smoothed_score: temporally smoothed smile score to reduce jitter
        self.face_tracking = {}
        self.face_id_counter = 0
        self.smoothing_factor = 0.7  # Higher = more smoothing (0.7 means 70% old, 30% new)
        
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
        Calculate smile score based on mouth region analysis.
        Uses multiple features: mouth width-to-height ratio, corner positions, and curvature analysis.
        More conservative scoring to reduce false positives.
        """
        h, w = image.shape[:2]
        
        # Extract face bounding box coordinates
        x_min, y_min, face_width, face_height = face_bbox
        x_max = x_min + face_width
        y_max = y_min + face_height
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min or face_width < 30 or face_height < 30:
            return 0.3  # Default to lower score if face too small
        
        # Extract face region
        face_region = image[y_min:y_max, x_min:x_max]
        
        if face_region.size == 0:
            return 0.3
        
        # Estimate mouth region (typically in lower 1/3 of face, centered horizontally)
        face_height_px = y_max - y_min
        face_width_px = x_max - x_min
        
        # Mouth region: lower portion of face
        mouth_y_start = int(face_height_px * 0.55)
        mouth_y_end = int(face_height_px * 0.90)
        mouth_x_start = int(face_width_px * 0.20)
        mouth_x_end = int(face_width_px * 0.80)
        
        if mouth_y_end <= mouth_y_start or mouth_x_end <= mouth_x_start:
            return 0.3
        
        mouth_region = face_region[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        
        if mouth_region.size == 0:
            return 0.3
        
        # Convert to grayscale
        if len(mouth_region.shape) == 3:
            mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            mouth_gray = mouth_region
        
        # Apply Gaussian blur to reduce noise
        mouth_gray = cv2.GaussianBlur(mouth_gray, (5, 5), 0)
        
        # Calculate mouth dimensions
        mouth_height = mouth_y_end - mouth_y_start
        mouth_width = mouth_x_end - mouth_x_start
        
        if mouth_height == 0 or mouth_width == 0:
            return 0.3
        
        # Feature 1: Mouth width-to-height ratio
        # When smiling, mouth is wider relative to height
        width_ratio = mouth_width / mouth_height
        # Normal neutral mouth: ratio ~1.5-2.5, smiling: ratio ~2.5-4.0
        width_score = np.clip((width_ratio - 1.5) / 2.5, 0.0, 1.0)
        
        # Feature 2: Analyze mouth curvature using histogram
        # Smiling mouths have more pixels in upper portion (teeth/curved shape)
        mouth_h, mouth_w = mouth_gray.shape
        upper_half = mouth_gray[:mouth_h//2, :]
        lower_half = mouth_gray[mouth_h//2:, :]
        
        # Calculate mean brightness (smiles often show teeth = brighter upper half)
        upper_mean = np.mean(upper_half) if upper_half.size > 0 else 0
        lower_mean = np.mean(lower_half) if lower_half.size > 0 else 0
        
        # When smiling, upper half tends to be brighter (teeth visible)
        brightness_score = 0.0
        if upper_half.size > 0 and lower_half.size > 0:
            brightness_diff = (upper_mean - lower_mean) / 255.0
            brightness_score = np.clip(brightness_diff * 2.0, 0.0, 0.5)  # Max 0.5 contribution
        
        # Feature 3: Horizontal edge detection for curvature
        # Apply Sobel operator to detect horizontal edges (smile curve)
        sobel_x = cv2.Sobel(mouth_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        
        # Focus on upper portion for curvature analysis
        upper_sobel = sobel_x[:mouth_h//2, :] if mouth_h > 4 else sobel_x
        horizontal_edge_strength = np.mean(upper_sobel) if upper_sobel.size > 0 else 0
        
        # Normalize edge strength
        edge_score = np.clip(horizontal_edge_strength / 50.0, 0.0, 0.4)  # Max 0.4 contribution
        
        # Feature 4: Corner analysis using edge detection
        # Smiling mouths have upward-curving corners
        corners = cv2.goodFeaturesToTrack(mouth_gray, maxCorners=10, qualityLevel=0.01, minDistance=10)
        corner_score = 0.0
        if corners is not None and len(corners) > 0:
            # If we detect corners (which happens more when smiling), add small score
            corner_score = min(len(corners) / 10.0, 0.2)  # Max 0.2 contribution
        
        # Feature 5: Mouth opening analysis
        # Smiling mouths are slightly open (height slightly increases)
        # Closed mouth when neutral, slightly open when smiling
        opening_score = 0.0
        if mouth_height > face_height_px * 0.15:  # Mouth is noticeably open
            opening_score = 0.1  # Small positive contribution
        
        # Combine features with weights (more conservative)
        # Require multiple indicators to score high
        smile_score = (
            width_score * 0.35 +      # Width ratio is important
            brightness_score * 0.25 +  # Teeth visibility
            edge_score * 0.20 +       # Curvature
            corner_score * 0.10 +      # Edge features
            opening_score * 0.10       # Opening
        )
        
        # Make scoring more conservative - require stronger signals
        smile_score = smile_score * 1.2  # Slight boost, but still conservative
        smile_score = np.clip(smile_score, 0.0, 1.0)
        
        # Apply threshold scaling - scores below 0.4 are unlikely to be smiles
        if smile_score < 0.4:
            smile_score = smile_score * 0.5  # Reduce low scores further
        
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
        
        for bbox in detected_faces:
            # Get face bounding box from OpenCV (x, y, width, height)
            x_min, y_min, face_width, face_height = bbox
            
            # Calculate face center for tracking (normalized coordinates for consistency)
            h, w = image.shape[:2]
            face_center_x = (x_min + face_width / 2) / w
            face_center_y = (y_min + face_height / 2) / h
            
            # Find closest existing face or create new one
            best_match_id = None
            min_distance = float('inf')
            
            for face_id, face_data in self.face_tracking.items():
                # Handle both old format (5 items) and new format (6 items with smoothed_score)
                x, y = face_data[0], face_data[1]
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
                face_data = self.face_tracking[best_match_id]
                if len(face_data) >= 5:
                    below_threshold_since = face_data[4]  # below_threshold_since is at index 4
                else:
                    below_threshold_since = None
            
            # Calculate smile score for this face
            raw_smile_score = self.calculate_smile_score(bbox, image)
            
            # Apply temporal smoothing to reduce jitter
            if best_match_id in self.face_tracking:
                old_smoothed_score = self.face_tracking[best_match_id][5] if len(self.face_tracking[best_match_id]) > 5 else raw_smile_score
                smoothed_score = (self.smoothing_factor * old_smoothed_score + (1 - self.smoothing_factor) * raw_smile_score)
            else:
                smoothed_score = raw_smile_score
            
            # Use smoothed score for threshold checking
            smile_score = smoothed_score
            
            # Update below_threshold_since tracking
            if smile_score < self.smile_threshold:
                if below_threshold_since is None:
                    # First time dropping below threshold
                    below_threshold_since = current_time
            else:
                # Above threshold, reset
                below_threshold_since = None
            
            faces_seen.add(best_match_id)
            self.face_tracking[best_match_id] = (face_center_x, face_center_y, smile_score, current_time, below_threshold_since, smoothed_score)
        
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
        
        for face_id, face_data in self.face_tracking.items():
            # Handle both old format (5 items) and new format (6 items with smoothed_score)
            if len(face_data) == 6:
                x, y, score, last_time, below_threshold_since, smoothed_score = face_data
            else:
                x, y, score, last_time, below_threshold_since = face_data[:5]
                smoothed_score = score
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
            scores = []
            for face_data in self.face_tracking.values():
                if len(face_data) >= 3:
                    score = face_data[2]  # smile_score is at index 2
                    if score is not None:
                        scores.append(score)
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
        current_time = time.time()
        h, w = image.shape[:2]
        
        # Convert to grayscale for face detection (Haar Cascade works on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        # scaleFactor: how much the image size is reduced at each scale
        # minNeighbors: how many neighbors each candidate rectangle should have
        # minSize: minimum possible object size
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Track faces and calculate smile scores
        if len(faces) > 0:
            # Update face tracking and calculate smile scores in one pass
            self.update_face_tracking(faces, image, current_time)
            
            # Draw bounding boxes and smile scores for each detected face
            for idx, bbox in enumerate(faces):
                x_min, y_min, face_width, face_height = bbox
                x_max = x_min + face_width
                y_max = y_min + face_height
                
                # Get face center for matching display (normalized)
                face_center_x = (x_min + face_width / 2) / w
                face_center_y = (y_min + face_height / 2) / h
                
                # Find corresponding face ID in tracking
                best_match_id = None
                min_distance = float('inf')
                for face_id, face_data in self.face_tracking.items():
                    x, y = face_data[0], face_data[1]
                    distance = math.hypot(x - face_center_x, y - face_center_y)
                    if distance < min_distance and distance < 0.1:
                        min_distance = distance
                        best_match_id = face_id
                
                # Get smile score for this face
                if best_match_id is not None:
                    face_data = self.face_tracking[best_match_id]
                    smile_score = face_data[2] if len(face_data) >= 3 else 0.3
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
    # Suppress Qt Wayland warning (doesn't affect functionality)
    import os
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
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

