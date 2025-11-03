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
        self.smile_threshold = 0.50  # Higher threshold - more strict, reduce false positives
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
        Calculate smile score based on mouth curvature and shape analysis.
        Focuses on detecting actual smiles (upward-curving mouth) rather than just mouth opening.
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
            return 0.2  # Default to low score if face too small
        
        # Extract face region
        face_region = image[y_min:y_max, x_min:x_max].copy()
        
        if face_region.size == 0:
            return 0.2
        
        # Estimate mouth region (typically in lower 1/3 of face, centered horizontally)
        face_height_px = y_max - y_min
        face_width_px = x_max - x_min
        
        # Mouth region: lower portion of face (narrower region focused on mouth)
        mouth_y_start = int(face_height_px * 0.60)
        mouth_y_end = int(face_height_px * 0.85)
        mouth_x_start = int(face_width_px * 0.25)
        mouth_x_end = int(face_width_px * 0.75)
        
        if mouth_y_end <= mouth_y_start or mouth_x_end <= mouth_x_start:
            return 0.2
        
        mouth_region = face_region[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        
        if mouth_region.size == 0:
            return 0.2
        
        # Convert to grayscale
        if len(mouth_region.shape) == 3:
            mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            mouth_gray = mouth_region
        
        # Apply Gaussian blur to reduce noise
        mouth_gray = cv2.GaussianBlur(mouth_gray, (5, 5), 0)
        
        mouth_h, mouth_w = mouth_gray.shape
        if mouth_h < 10 or mouth_w < 20:
            return 0.2
        
        # Feature 1: Detect upward curvature using edge analysis
        # Smiles have upward-curving edges, neutral mouths are more horizontal
        # Use horizontal gradient to detect the smile curve
        
        # Apply horizontal gradient (Sobel X) to detect vertical edges
        sobel_x = cv2.Sobel(mouth_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        
        # Apply vertical gradient (Sobel Y) to detect horizontal edges (smile curve)
        sobel_y = cv2.Sobel(mouth_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        
        # For a smile, we expect:
        # - Strong horizontal gradients in the middle/upper portion (smile curve)
        # - The curve should be upward (higher gradient in upper half)
        
        # Divide mouth into thirds: left, center, right
        left_third = sobel_y[:, :mouth_w//3]
        center_third = sobel_y[:, mouth_w//3:2*mouth_w//3]
        right_third = sobel_y[:, 2*mouth_w//3:]
        
        # Upper half (where smile curve is most visible)
        upper_half = mouth_gray[:mouth_h//2, :]
        lower_half = mouth_gray[mouth_h//2:, :]
        
        # Calculate curvature score based on gradient distribution
        # In a smile, horizontal gradients (sobel_y) should be stronger in the upper portion
        upper_sobel_y = sobel_y[:mouth_h//2, :] if mouth_h > 4 else sobel_y
        lower_sobel_y = sobel_y[mouth_h//2:, :] if mouth_h > 4 else sobel_y
        
        upper_gradient = np.mean(upper_sobel_y) if upper_sobel_y.size > 0 else 0
        lower_gradient = np.mean(lower_sobel_y) if lower_sobel_y.size > 0 else 0
        
        # Smile has stronger gradients in upper portion (upward curve)
        # Made stricter - require more pronounced curvature
        curvature_score = 0.0
        if upper_gradient > 0:
            gradient_ratio = upper_gradient / (upper_gradient + lower_gradient + 1e-6)
            # More strict: require clearer gradient difference (higher threshold)
            curvature_score = np.clip((gradient_ratio - 0.45) * 2.5, 0.0, 1.0)  # Stricter threshold
        # Only give minimum score if gradient difference is very significant
        if abs(upper_gradient - lower_gradient) > 15:  # Increased from 5 to 15
            curvature_score = max(curvature_score, 0.2)  # Lower minimum score
        
        # Feature 2: Analyze mouth shape using contour detection
        # Smiling mouths have a distinct upward-curving contour
        edges = cv2.Canny(mouth_gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_score = 0.0
        if len(contours) > 0:
            # Find the largest contour (likely the mouth outline)
            largest_contour = max(contours, key=cv2.contourArea)
            # Lower threshold for contour area to be more lenient
            if cv2.contourArea(largest_contour) > mouth_w * mouth_h * 0.05:  # Reduced from 0.1
                # Fit a curve/ellipse to the contour
                if len(largest_contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest_contour)
                        # Get ellipse center and axes
                        center, axes, angle = ellipse
                        major_axis, minor_axis = max(axes), min(axes)
                        
                        # For a smile, the ellipse should be wider (major axis horizontal)
                        # and angled upward (negative angle indicates upward curve)
                        if major_axis > 0:
                            axis_ratio = minor_axis / major_axis
                            # Wider ellipses (smaller ratio) suggest a smile - more lenient
                            shape_score = np.clip((0.7 - axis_ratio) * 2.5, 0.0, 0.6)  # More lenient, higher max
                    except:
                        # If ellipse fitting fails, still give some score for having a contour
                        shape_score = 0.2
            else:
                # Even if contour is small, if there are multiple contours, it might indicate a smile
                if len(contours) >= 2:
                    shape_score = 0.15
        
        # Feature 3: Corner elevation analysis
        # Smiling mouths have corners that are elevated relative to center
        # Analyze the vertical profile of the mouth
        
        # Get horizontal line profiles at different vertical positions
        top_row = mouth_gray[0, :]
        middle_row = mouth_gray[mouth_h//2, :]
        bottom_row = mouth_gray[-1, :]
        
        # Calculate the "curvature" by comparing corner positions
        # In a smile, corners should be higher than the center
        left_corner_top = np.mean(mouth_gray[:mouth_h//3, :mouth_w//4])
        left_corner_bottom = np.mean(mouth_gray[2*mouth_h//3:, :mouth_w//4])
        right_corner_top = np.mean(mouth_gray[:mouth_h//3, 3*mouth_w//4:])
        right_corner_bottom = np.mean(mouth_gray[2*mouth_h//3:, 3*mouth_w//4:])
        center_top = np.mean(mouth_gray[:mouth_h//3, mouth_w//3:2*mouth_w//3])
        center_bottom = np.mean(mouth_gray[2*mouth_h//3:, mouth_w//3:2*mouth_w//3])
        
        # In a smile, corners are elevated (brighter/different) and center may dip
        corner_elevation = 0.0
        if center_top > 0 and center_bottom > 0:
            left_elevation = abs(left_corner_top - left_corner_bottom) / 255.0
            right_elevation = abs(right_corner_top - right_corner_bottom) / 255.0
            avg_elevation = (left_elevation + right_elevation) / 2.0
            # More sensitive to elevation differences
            corner_elevation = np.clip(avg_elevation * 4.0, 0.0, 0.7)  # Increased sensitivity and max
        
        # Also give score if corners are simply different from center (even if not elevated)
        if center_top > 0:
            corner_diff_left = abs(left_corner_top - center_top) / 255.0
            corner_diff_right = abs(right_corner_top - center_top) / 255.0
            avg_corner_diff = (corner_diff_left + corner_diff_right) / 2.0
            corner_elevation = max(corner_elevation, np.clip(avg_corner_diff * 2.0, 0.0, 0.4))
        
        # Feature 4: Width increase (but NOT due to opening)
        # Smiles widen the mouth without necessarily opening it much
        mouth_width = mouth_x_end - mouth_x_start
        mouth_height = mouth_y_end - mouth_y_start
        
        # Calculate aspect ratio, but penalize if mouth is too open (not a smile, just opening)
        width_ratio = mouth_width / mouth_height if mouth_height > 0 else 1.0
        
        # Smile should have good width ratio but not be too open
        # Made stricter - penalize excessive opening (likely not a smile)
        width_score = 0.0
        if 2.0 <= width_ratio <= 3.5:  # Narrower ideal range
            # Good range for smiles
            width_score = np.clip((width_ratio - 1.8) / 1.7, 0.0, 0.4)  # Lower max score
        elif 3.5 < width_ratio <= 4.5:
            # Acceptable but not ideal (might be opening mouth)
            width_score = np.clip((4.5 - width_ratio) / 1.0, 0.0, 0.2)  # Lower score
        elif width_ratio > 4.5:
            # Too open - likely just opening mouth, heavily penalize
            width_score = 0.0
        else:
            # Too narrow - give minimal score
            width_score = np.clip(width_ratio / 3.0, 0.0, 0.25)  # Lower max for narrow
        
        # Combine features with emphasis on curvature and shape (not opening)
        # Made stricter - require multiple strong indicators
        smile_score = (
            curvature_score * 0.40 +    # Curvature is most important (40%)
            shape_score * 0.30 +         # Shape analysis (30%)
            corner_elevation * 0.20 +    # Corner elevation (20%)
            width_score * 0.10           # Width - reduced weight (10%)
        )
        
        # Stricter bonus - require multiple STRONG indicators, not just any indicators
        active_features = sum([1 for score in [curvature_score, shape_score, corner_elevation, width_score] if score > 0.4])  # Higher threshold
        if active_features >= 3:  # Require 3 instead of 2
            smile_score += 0.1  # Smaller bonus
        elif active_features >= 2:
            smile_score += 0.05  # Very small bonus
        
        # Less aggressive boost - more conservative
        smile_score = smile_score * 1.2  # Reduced from 1.5
        smile_score = np.clip(smile_score, 0.0, 1.0)
        
        # Apply penalty for scores that aren't clearly smiles
        if smile_score < 0.4:
            smile_score = smile_score * 0.7  # More aggressive penalty for low scores
        
        # Additional check: if width_score is high but curvature is low, it's probably just opening mouth, not smiling
        if width_score > 0.5 and curvature_score < 0.3:
            smile_score *= 0.6  # Penalize mouth opening without curvature
        
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
                
                # Determine if this person is smiling
                is_smiling = smile_score >= self.smile_threshold
                
                # Draw bounding box with different styles for smiling vs not smiling
                if is_smiling:
                    # Green outline for smiling faces (thinner, less prominent)
                    box_color = (0, 255, 0)
                    thickness = 2
                    # Draw the bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, thickness)
                    
                    # Display smile score
                    face_label = best_match_id if best_match_id is not None else idx
                    score_text = f"Face {face_label}: {smile_score:.2f}"
                    cv2.putText(image, score_text, (x_min, max(y_min - 10, 20)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                else:
                    # RED OUTLINE for non-smiling faces - make it very obvious
                    box_color = (0, 0, 255)  # Bright red
                    thickness = 5  # Thick red outline
                    
                    # Draw multiple red rectangles for emphasis (glow effect)
                    for offset in range(3, 0, -1):
                        cv2.rectangle(image, 
                                     (x_min-offset, y_min-offset), 
                                     (x_max+offset, y_max+offset), 
                                     (0, 0, 255), 2)
                    
                    # Main thick red bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, thickness)
                    
                    # Draw diagonal corner markers for extra visibility
                    corner_size = 15
                    # Top-left corner
                    cv2.line(image, (x_min, y_min), (x_min + corner_size, y_min), box_color, 3)
                    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_size), box_color, 3)
                    # Top-right corner
                    cv2.line(image, (x_max, y_min), (x_max - corner_size, y_min), box_color, 3)
                    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_size), box_color, 3)
                    # Bottom-left corner
                    cv2.line(image, (x_min, y_max), (x_min + corner_size, y_max), box_color, 3)
                    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_size), box_color, 3)
                    # Bottom-right corner
                    cv2.line(image, (x_max, y_max), (x_max - corner_size, y_max), box_color, 3)
                    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_size), box_color, 3)
                    
                    # Display warning message with background
                    face_label = best_match_id if best_match_id is not None else idx
                    warning_text = f"Face {face_label}: NOT SMILING!"
                    score_text = f"Score: {smile_score:.2f}"
                    
                    # Calculate text size and position
                    text_size_warning = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_size_score = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    text_x = x_min
                    text_y_warning = max(y_min - 20, 30)
                    text_y_score = text_y_warning + text_size_warning[1] + 8
                    
                    # Background rectangle for warning text (red background)
                    cv2.rectangle(image, 
                                 (text_x - 5, text_y_warning - text_size_warning[1] - 5),
                                 (text_x + text_size_warning[0] + 5, text_y_warning + 5),
                                 (0, 0, 255), -1)
                    
                    # Background rectangle for score text
                    cv2.rectangle(image, 
                                 (text_x - 5, text_y_score - text_size_score[1] - 5),
                                 (text_x + text_size_score[0] + 5, text_y_score + 5),
                                 (0, 0, 200), -1)
                    
                    # Warning text (white on red)
                    cv2.putText(image, warning_text, (text_x, text_y_warning),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Score text
                    cv2.putText(image, score_text, (text_x, text_y_score),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw red indicator circle at top center of face
                    center_x = (x_min + x_max) // 2
                    cv2.circle(image, (center_x, y_min - 12), 10, (0, 0, 255), -1)
                    cv2.circle(image, (center_x, y_min - 12), 10, (255, 255, 255), 2)
                    # Add exclamation mark
                    cv2.putText(image, "!", (center_x - 5, y_min - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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

