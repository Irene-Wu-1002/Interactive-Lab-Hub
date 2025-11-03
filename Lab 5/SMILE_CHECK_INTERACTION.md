# Multi-Person Smile-Check Camera - Interaction Documentation

## Interaction Concept

The **Multi-Person Smile-Check Camera** is a playful observant system designed for group photo scenarios. It acts as an automated "smile checker" that monitors multiple faces simultaneously and provides gentle feedback when not everyone in the frame is smiling.

### Core Interaction Loop

1. **Detection**: The system continuously detects multiple faces in the webcam feed using MediaPipe Face Mesh
2. **Assessment**: Each face is independently evaluated for a "smile score" using facial landmark analysis
3. **Monitoring**: The system tracks whether all detected faces maintain a smile above the threshold (0.65)
4. **Feedback**: When any face falls below the threshold for more than 1 second, the system triggers multimodal feedback to nudge the group toward smiling

## Technical Implementation

### Face Detection & Tracking
- **Model**: MediaPipe Face Mesh (uses BlazeFace-based detection under the hood)
- **Capacity**: Up to 5 simultaneous faces
- **Tracking**: Simple distance-based face matching between frames (prototype stage)
- **Landmarks**: Uses 468 facial landmarks provided by MediaPipe Face Mesh

### Smile Estimation (Prototype Heuristic)
The current implementation uses a simple geometric heuristic based on mouth landmarks:
- **Mouth Width-to-Height Ratio**: When smiling, the mouth typically widens more than it opens vertically
- **Normalization**: Ratios are mapped to a 0-1 smile score scale
- **Threshold**: Scores below 0.65 trigger feedback

**Note**: This is a prototype heuristic. In production, this would be replaced with:
- A trained smile classifier (e.g., CNN trained on smile/no-smile datasets)
- Fine-tuned landmark-based models
- Transfer learning from existing emotion recognition models

### Feedback Modes

The system experiments with multiple simultaneous feedback channels:

#### 1. **Audio Feedback**
- Soft "ding" tone (440 Hz beep, 0.2 seconds)
- Plays once when feedback activates
- Fallback: System beep if pygame not available

#### 2. **Visual LED Simulation**
- Cyan LED indicator in top-right corner
- Blinks at 0.5 Hz (2 blinks per second) when feedback active
- Can be easily adapted to control physical LEDs via GPIO

#### 3. **Screen Overlay**
- **Red Border**: 10-pixel red border around entire frame when feedback active
- **Warning Text**: "Someone isn't smiling yet!" displayed at bottom center
- **Status Bar**: Real-time smile progress bar at top showing minimum smile score across all faces

#### 4. **Face-Level Feedback**
- Each detected face gets a bounding box
- **Green box**: Smile score ≥ 0.65 (smiling)
- **Red box**: Smile score < 0.65 (not smiling)
- Individual smile scores displayed above each face

## Interaction Flow

```
┌─────────────────┐
│ Camera Captures │
│   Frame         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detect Faces    │
│ (MediaPipe)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calculate Smile │
│ Score per Face  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ All Smiling?    │ YES  │  Calm State  │
│ (score ≥ 0.65)  │─────▶│  No Feedback │
└────────┬────────┘      └──────────────┘
         │ NO
         ▼
┌─────────────────┐
│ Below Threshold │
│ for >1 second?  │
└────────┬────────┘
         │ YES
         ▼
┌─────────────────┐
│ Trigger Multi-  │
│ modal Feedback  │
└─────────────────┘
```

## Experimentation Notes

### What Works Well
1. **Real-time Performance**: MediaPipe Face Mesh runs efficiently at 30+ FPS on standard hardware
2. **Multi-face Detection**: Successfully tracks 2-5 faces simultaneously
3. **Visual Feedback**: The red border and warning text provide clear, immediate feedback
4. **Progress Bar**: Users can see their collective smile status at a glance

### Limitations & Edge Cases
1. **Smile Heuristic**: The geometric ratio approach has limitations:
   - False positives: Wide mouths (yawning, talking) may register as smiles
   - False negatives: Subtle smiles or asymmetric expressions may score low
   - Lighting conditions affect landmark detection accuracy

2. **Face Tracking**: Simple distance-based matching can fail when:
   - Faces move quickly
   - Faces overlap or occlude
   - Faces enter/exit frame rapidly

3. **Occlusion**: Partial face visibility reduces accuracy

4. **Angle Dependency**: Profile views or extreme angles reduce detection reliability

### Potential Improvements
1. **Better Smile Detection**:
   - Train a dedicated smile classifier using transfer learning
   - Use more sophisticated landmark-based features (e.g., lip corner displacement, mouth opening ratio)
   - Add temporal smoothing to reduce jitter

2. **Robust Face Tracking**:
   - Implement Kalman filtering for face position tracking
   - Use face recognition embeddings for identity persistence
   - Handle face entry/exit more gracefully

3. **Adaptive Thresholds**:
   - Personalize thresholds per individual
   - Learn from user behavior over time
   - Adjust based on lighting conditions

4. **Enhanced Feedback**:
   - Gradient feedback (getting closer to threshold)
   - Different feedback for different people (e.g., "Person 1, smile!")
   - Countdown timer for photo capture once all are smiling

## Usage

```bash
# Run the application
python smile_check_camera.py

# Press 'q' to quit
```

**Requirements**:
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- (Optional) PyGame for enhanced audio feedback

## Design Considerations

### Social Interaction
- **Non-intrusive**: Feedback is gentle (soft beep, visual cues) rather than aggressive
- **Cooperative**: Encourages group coordination ("we all need to smile")
- **Playful**: The concept invites experimentation and engagement

### Privacy & Ethics
- **Local Processing**: All face detection happens locally, no data sent to servers
- **Temporary**: No face data is stored or recorded
- **Consent**: Designed for contexts where users are aware of camera usage (e.g., photo booths)

### Accessibility
- Visual feedback may not be sufficient for visually impaired users
- Audio feedback helps but could be enhanced with haptics
- Consider alternative feedback modes for different sensory needs

## Future Directions

1. **Contextual Adaptation**: Adjust thresholds based on cultural norms or photo type (formal vs. casual)
2. **Multi-modal Integration**: Combine with audio analysis (detect laughter) or body pose (detect excitement)
3. **Gamification**: Add scoring, achievements, or social sharing features
4. **Physical Device**: Integrate with Raspberry Pi LEDs, buzzers, or display screens
5. **Smart Capture**: Automatically capture photo once all faces meet threshold for sustained period

