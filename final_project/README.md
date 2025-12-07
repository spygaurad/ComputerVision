# Table Tennis Ball Tracking System
## Complete Technical Documentation

---

## What Does This System Do?

This system takes a video of a table tennis game and:
1. Detects the ball in every frame
2. Draws a bounding box around it
3. Calculates and displays the ball's speed in km/h
4. Outputs a new video with all the tracking visualizations

---

## How It Works: The Pipeline

```
Video Frame → Pre-processing → Color Detection → Motion Detection → Candidate Filtering → Scoring → Best Match → Speed Calculation → Draw & Output
```

---

## Step 1: Calibration

### The Problem
Pixels don't mean anything in the real world. We need to know "how many pixels = 1 meter" to calculate actual speed.

### The Solution
We use the table tennis table as a reference since it has known dimensions:

| Dimension | Real Size |
|-----------|-----------|
| Length | 2.74 meters |
| Width | 1.525 meters |

### How It Works
1. User clicks 4 corners of the table in the first frame
2. System calculates pixel distance between corners
3. Divides by real-world distance to get **pixels-per-meter ratio**

```python
pixels_per_meter = table_length_in_pixels / 2.74
```

### Bonus: Region of Interest (ROI)
During calibration, we also set up an ROI - a rectangle around the table area. This tells the system "only look for the ball in this region", which:
- Ignores background distractions
- Speeds up processing
- Reduces false detections

---

## Step 2: Color Detection

### The Problem
We need to find orange pixels that belong to the ball, not orange shirts, signs, or walls.

### The Solution: HSV Color Space

We convert the image from BGR to HSV because:
- **BGR**: Red, Green, Blue mixed together - hard to isolate "orange"
- **HSV**: Hue (color), Saturation (intensity), Value (brightness) - easy to say "give me orange"

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

### HSV Values for Orange Ball

| Channel | Min | Max | Why |
|---------|-----|-----|-----|
| Hue | 5 | 20 | Orange is in this range (0-180 scale) |
| Saturation | 150 | 255 | High = vivid color, filters out pale objects |
| Value | 150 | 255 | High = bright, filters out dark objects |

### The Trick: High Saturation Minimum
Setting saturation minimum to 150 (not 100) was crucial. This filtered out:
- Faded signs on walls
- Skin tones
- Wooden furniture
- Other "orangish" but not vivid objects

---

## Step 3: Cleaning Up the Mask

### The Problem
The color mask has noise - tiny dots and holes.

### The Solution: Morphological Operations

**Opening** (Erode then Dilate):
- Removes small white dots (noise)
- Like "minimum size filter"

**Closing** (Dilate then Erode):
- Fills small holes inside the ball
- Connects nearby pixels

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
```

### The Trick: Elliptical Kernel
We use an ellipse-shaped kernel instead of a square because:
- Ball is round
- Ellipse preserves round shapes better
- Square kernels can make circles look blocky

---

## Step 4: Finding Candidates

### The Problem
After masking, we might have multiple white blobs. Which one is the ball?

### The Solution: Contour Analysis
We find all white regions (contours) and analyze each one.

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### Filters We Apply

| Filter | What It Checks | Ball Should Be |
|--------|----------------|----------------|
| Area | Size in pixels² | 150 - 8000 px² |
| Aspect Ratio | Width / Height | 0.3 - 3.5 (allows motion blur) |
| Circularity | How round | > 0.3 (fast) or > 0.5 (slow) |
| Solidity | Area / Convex Hull | > 0.4 (no holes) |

### Circularity Formula
```
Circularity = 4π × Area / Perimeter²
```
- Perfect circle = 1.0
- Square ≈ 0.78
- Elongated shape < 0.5

---

## Step 5: Handling Fast Balls (The Hard Part)

### Challenge 1: Motion Blur

**Problem**: Fast ball becomes a streak, not a circle. Circularity filter rejects it.

**Solution**: Adaptive thresholds based on speed.

```python
if ball_is_moving_fast:
    min_circularity = 0.3   # Very relaxed
    min_aspect_ratio = 0.3  # Allow elongated
    max_aspect_ratio = 3.5
else:
    min_circularity = 0.5   # Stricter
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2.0
```

### Challenge 2: Ball Moves Too Far Between Frames

**Problem**: At 60 FPS, a 100 km/h ball moves ~46 pixels per frame. Our search window was too small.

**Solution**: Increased `max_ball_speed_pixels` from 200 to 500.

### Challenge 3: Motion Blur Changes Ball Color

**Problem**: Blurred ball is lighter/more transparent, fails color detection.

**Solution**: Frame differencing - detect motion instead of color.

```python
# Compare current frame to previous frame
frame_diff = cv2.absdiff(previous_gray, current_gray)
_, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
```

This creates a mask of "things that moved" regardless of color.

### Challenge 4: Motion-Blurred Shape Detection

**Problem**: Elongated blur could be anything - how do we know it's the ball?

**Solution**: Check if elongation aligns with ball's velocity direction.

```python
# Get angle of the elongated shape
ellipse = cv2.fitEllipse(contour)
contour_angle = ellipse[2]

# Get angle of ball's velocity
velocity_angle = arctan2(velocity_y, velocity_x)

# If angles match (within 45°), it's probably the ball
if angle_difference < 45°:
    accept_as_ball()
```

---

## Step 6: Scoring Candidates

### The Problem
Multiple candidates pass all filters. Which is the real ball?

### The Solution: Weighted Scoring System

Each candidate gets a score from 0 to 1 based on:

| Factor | Weight | What It Rewards |
|--------|--------|-----------------|
| Shape | 25% | Circular, solid shapes |
| Proximity | 45% | Close to predicted position |
| Size Consistency | 30% | Similar size to recent detections |

### Proximity Scoring (Most Important)

We don't just check distance to last position - we check distance to **predicted** position:

```python
predicted_x = last_x + velocity_x
predicted_y = last_y + velocity_y

distance_to_prediction = sqrt((candidate_x - predicted_x)² + (candidate_y - predicted_y)²)
```

This helps during fast rallies when the ball moves significantly between frames.

### Bonus Multipliers

| Condition | Multiplier | Why |
|-----------|------------|-----|
| Motion-aligned shape | 1.2x | Likely a motion-blurred ball |
| Color detection (vs motion) | 1.0x vs 0.8x | Color is more reliable |

---

## Step 7: Prediction When Ball Is Lost

### The Problem
Sometimes the ball is:
- Behind a player
- Too blurry to detect
- Outside the frame momentarily

### The Solution: Velocity-Based Prediction

When detection fails, we predict where the ball should be:

```python
predicted_x = last_x + velocity_x × decay
predicted_y = last_y + velocity_y × decay
```

### The Trick: Decay Factor
We apply slight deceleration (0.95 per frame) because:
- Ball naturally slows down
- Prevents prediction from "running away"
- More realistic trajectory

### Prediction Limits

| Ball Speed | Max Prediction Frames |
|------------|----------------------|
| Slow | 10 frames |
| Fast | 15 frames |

After this, we reset and wait for a fresh detection.

---

## Step 8: Speed Calculation

### Converting Pixels to Meters
```python
distance_meters = distance_pixels / pixels_per_meter
```

### Multi-Frame Smoothing

Instead of frame-to-frame speed (very noisy), we calculate over 5 frames:

```python
lookback = 5
distance = position[now] - position[now - lookback]
time = timestamps[now] - timestamps[now - lookback]
speed = distance / time
```

### Rolling Average

We keep last 10 speed values and average them:

```python
speed_history.append(current_speed)
smoothed_speed = mean(speed_history)
```

This prevents the display from jumping around wildly.

### Unit Conversion
```python
speed_kmh = speed_mps × 3.6
```

---

## Summary: All Techniques Used

| Challenge | Technique | Result |
|-----------|-----------|--------|
| Pixel-to-meter conversion | Table calibration | Accurate speed |
| Background distractions | ROI filtering | Fewer false positives |
| Finding orange ball | HSV color space | Robust to lighting |
| Noise in mask | Morphological operations | Clean detection |
| Non-ball orange objects | High saturation threshold | Filters signs/skin |
| Multiple candidates | Multi-criteria scoring | Picks best match |
| Ball moves between frames | Proximity to prediction | Tracks fast motion |
| Motion blur (shape) | Adaptive circularity | Accepts elongated shapes |
| Motion blur (color) | Frame differencing | Detects by movement |
| Blur direction | Velocity alignment check | Validates blur is ball |
| Temporary occlusion | Position prediction | Maintains tracking |
| Noisy speed readings | Multi-frame smoothing | Stable display |

---

## Usage Quick Reference

### Basic
```bash
python ball_tracker.py video.mp4
```

### Custom Output
```bash
python ball_tracker.py video.mp4 -o output.mp4
```

### Different Ball Color
```bash
python ball_tracker.py video.mp4 --ball-color white
```

### Tune Detection (Opens Sliders)
```bash
python ball_tracker.py video.mp4 --tune
```

### Adjust Ball Size
```bash
python ball_tracker.py video.mp4 --min-area 100 --max-area 3000
```

### Show Trajectory Trail
```bash
python ball_tracker.py video.mp4 --trail
```

### Faster Processing
```bash
python ball_tracker.py video.mp4 --no-preview
```

---

## Output Video Contents

| Element | Color | Meaning |
|---------|-------|---------|
| Bounding box | Green | Confirmed detection |
| Bounding box | Orange | Predicted position |
| Center dot | Red | Ball center |
| Table outline | Purple | Calibrated area |
| Trail (if enabled) | Orange | Recent trajectory |

### Info Panel (Top Left)
- Frame counter
- Video FPS
- Current speed (km/h and m/s)
- Ball color mode
- Calibration scale

---

## Dependencies

```bash
pip install opencv-python numpy
```

---

## Future Improvements

1. **Deep Learning**: Use YOLO or similar for more robust detection
2. **Kalman Filter**: Smoother prediction with physics model
3. **3D Tracking**: Stereo cameras for true 3D position
4. **Spin Detection**: High-speed camera + rotation analysis
5. **Auto Table Detection**: No manual calibration needed