"""
Table Tennis Ball Tracker and Speed Calculator
Uses OpenCV to track a ball, draw bounding boxes, and display speed.
Outputs a video with tracking visualization.

Standard Butterfly Table Tennis Table Dimensions:
- Length: 2.74 meters (9 feet)
- Width: 1.525 meters (5 feet)
"""

import cv2
import numpy as np
from collections import deque
import argparse
import time


class TableTennisTracker:
    # Standard table dimensions in meters
    TABLE_LENGTH_M = 2.74
    TABLE_WIDTH_M = 1.525
    
    def __init__(self, video_path, output_path="output_tracked.mp4", buffer_size=64):
        self.video_path = video_path
        self.output_path = output_path
        self.buffer_size = buffer_size
        
        # Position and time tracking
        self.positions = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # Calibration variables
        self.pixels_per_meter = None
        self.table_corners = []
        self.calibrated = False
        
        # Ball detection parameters - HSV ranges
        # Orange table tennis ball - adjusted for typical indoor lighting
        self.hsv_ranges = {
            "orange": {
                "lower": np.array([5, 150, 150]),   # More restrictive saturation/value
                "upper": np.array([20, 255, 255])   # Narrower hue range
            },
            "white": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([180, 50, 255])
            },
            "yellow": {
                "lower": np.array([20, 100, 100]),
                "upper": np.array([35, 255, 255])
            }
        }
        
        self.ball_color = "orange"  # Default
        
        # Region of interest (set during calibration to focus on table area)
        self.roi = None  # Will be (x, y, w, h) or None for full frame
        self.use_roi = True  # Focus detection near the table
        
        # Ball size constraints (in pixels) - will be adjusted after calibration
        # Based on screenshot: ball appears ~20-40px diameter, so area ~300-1500
        self.min_ball_area = 150      # Minimum contour area (filters tiny noise)
        self.max_ball_area = 8000     # Maximum contour area (increased for motion blur)
        
        # Tracking continuity variables
        self.last_known_position = None
        self.velocity = (0, 0)
        self.frames_since_detection = 0
        self.max_prediction_frames = 10  # Increased for fast ball recovery
        self.max_ball_speed_pixels = 500  # Increased for fast shots!
        self.recent_ball_sizes = deque(maxlen=20)
        
        # Motion-aware detection
        self.current_speed_estimate = 0  # pixels per frame
        self.prev_frame_gray = None  # For frame differencing
        self.use_frame_differencing = True  # Enable motion detection
        
        # Speed calculation
        self.speed_history = deque(maxlen=10)
        self.current_speed_mps = 0
        self.current_speed_kmh = 0
        
        # Tracking trail (disabled by default now)
        self.show_trail = False
        self.trail_points = deque(maxlen=10)
        
        # Slow motion output options
        self.slow_mo_fps = None      # Output at specific FPS (e.g., 5)
        self.slow_mo_factor = None   # Slow down by factor (e.g., 4 = 4x slower)
        
    def calibrate_from_table(self, frame):
        """
        Manual calibration: click on the four corners of the table.
        """
        print("\n" + "="*50)
        print("CALIBRATION MODE")
        print("="*50)
        print("Click on the 4 corners of the table in order:")
        print("  1. Top-left corner")
        print("  2. Top-right corner")
        print("  3. Bottom-right corner")
        print("  4. Bottom-left corner")
        print("\nControls:")
        print("  'r' - Reset corners")
        print("  'c' - Confirm calibration (after 4 points)")
        print("  'q' - Quit calibration")
        print("="*50 + "\n")
        
        self.table_corners = []
        calibration_frame = frame.copy()
        display_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal display_frame
            if event == cv2.EVENT_LBUTTONDOWN and len(self.table_corners) < 4:
                self.table_corners.append((x, y))
                display_frame = calibration_frame.copy()
                
                # Draw all points and lines
                for i, pt in enumerate(self.table_corners):
                    cv2.circle(display_frame, pt, 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if i > 0:
                        cv2.line(display_frame, self.table_corners[i-1], pt, (0, 255, 0), 2)
                
                if len(self.table_corners) == 4:
                    cv2.line(display_frame, self.table_corners[3], self.table_corners[0], (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'c' to confirm", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow("Calibration", display_frame)
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        # Add instructions on frame
        cv2.putText(display_frame, "Click 4 table corners", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Calibration", display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.table_corners = []
                display_frame = calibration_frame.copy()
                cv2.putText(display_frame, "Click 4 table corners", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Calibration", display_frame)
            elif key == ord('c') and len(self.table_corners) == 4:
                break
            elif key == ord('q'):
                cv2.destroyWindow("Calibration")
                return False
        
        cv2.destroyWindow("Calibration")
        
        # Calculate pixels per meter from table edges
        top_edge = np.linalg.norm(
            np.array(self.table_corners[1]) - np.array(self.table_corners[0])
        )
        bottom_edge = np.linalg.norm(
            np.array(self.table_corners[2]) - np.array(self.table_corners[3])
        )
        avg_length_pixels = (top_edge + bottom_edge) / 2
        
        left_edge = np.linalg.norm(
            np.array(self.table_corners[3]) - np.array(self.table_corners[0])
        )
        right_edge = np.linalg.norm(
            np.array(self.table_corners[2]) - np.array(self.table_corners[1])
        )
        avg_width_pixels = (left_edge + right_edge) / 2
        
        # Use length for calibration (more reliable as it's longer)
        self.pixels_per_meter = avg_length_pixels / self.TABLE_LENGTH_M
        
        # Set up ROI (region of interest) around the table with padding
        # This helps filter out detections far from the playing area
        all_x = [pt[0] for pt in self.table_corners]
        all_y = [pt[1] for pt in self.table_corners]
        padding = 200  # Extra space above/around table for ball trajectory
        
        roi_x = max(0, min(all_x) - padding)
        roi_y = max(0, min(all_y) - padding * 2)  # More padding above for high balls
        roi_w = max(all_x) - min(all_x) + padding * 2
        roi_h = max(all_y) - min(all_y) + padding * 3
        
        self.roi = (roi_x, roi_y, roi_w, roi_h)
        
        self.calibrated = True
        print(f"\n✓ Calibration complete!")
        print(f"  Pixels per meter: {self.pixels_per_meter:.2f}")
        print(f"  Table length in pixels: {avg_length_pixels:.0f}")
        print(f"  Table width in pixels: {avg_width_pixels:.0f}")
        print(f"  ROI set to focus on table area\n")
        return True
    
    def set_ball_color(self, color):
        """Set the ball color for detection."""
        if color in self.hsv_ranges:
            self.ball_color = color
            print(f"Ball color set to: {color}")
        else:
            print(f"Unknown color. Available: {list(self.hsv_ranges.keys())}")
    
    def tune_ball_detection(self, frame):
        """
        Interactive tuning of HSV values for ball detection.
        """
        print("\n" + "="*50)
        print("BALL DETECTION TUNING")
        print("="*50)
        print("Adjust sliders to detect the ball (shown in white)")
        print("Press 'q' when done")
        print("="*50 + "\n")
        
        cv2.namedWindow("Tune Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        
        # Get current values
        current = self.hsv_ranges[self.ball_color]
        
        cv2.createTrackbar("H Low", "Tune Detection", int(current["lower"][0]), 180, lambda x: None)
        cv2.createTrackbar("H High", "Tune Detection", int(current["upper"][0]), 180, lambda x: None)
        cv2.createTrackbar("S Low", "Tune Detection", int(current["lower"][1]), 255, lambda x: None)
        cv2.createTrackbar("S High", "Tune Detection", int(current["upper"][1]), 255, lambda x: None)
        cv2.createTrackbar("V Low", "Tune Detection", int(current["lower"][2]), 255, lambda x: None)
        cv2.createTrackbar("V High", "Tune Detection", int(current["upper"][2]), 255, lambda x: None)
        
        while True:
            h_low = cv2.getTrackbarPos("H Low", "Tune Detection")
            h_high = cv2.getTrackbarPos("H High", "Tune Detection")
            s_low = cv2.getTrackbarPos("S Low", "Tune Detection")
            s_high = cv2.getTrackbarPos("S High", "Tune Detection")
            v_low = cv2.getTrackbarPos("V Low", "Tune Detection")
            v_high = cv2.getTrackbarPos("V High", "Tune Detection")
            
            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Find contours and draw
            display = frame.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow("Tune Detection", display)
            cv2.imshow("Mask", mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save the tuned values
        self.hsv_ranges[self.ball_color]["lower"] = np.array([h_low, s_low, v_low])
        self.hsv_ranges[self.ball_color]["upper"] = np.array([h_high, s_high, v_high])
        
        cv2.destroyWindow("Tune Detection")
        cv2.destroyWindow("Mask")
        
        print(f"Updated HSV range for {self.ball_color}:")
        print(f"  Lower: [{h_low}, {s_low}, {v_low}]")
        print(f"  Upper: [{h_high}, {s_high}, {v_high}]")
    
    def detect_ball(self, frame):
        """
        Detect the ball using combined color + motion detection.
        Adapts constraints based on estimated ball speed for fast ball tracking.
        Returns: (x, y, w, h, cx, cy) bounding box and center or None if not found
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Apply ROI if set
        roi_offset_x, roi_offset_y = 0, 0
        if self.use_roi and self.roi is not None:
            rx, ry, rw, rh = self.roi
            rx, ry = max(0, int(rx)), max(0, int(ry))
            rw = min(int(rw), frame_w - rx)
            rh = min(int(rh), frame_h - ry)
            working_frame = frame[ry:ry+rh, rx:rx+rw]
            roi_offset_x, roi_offset_y = rx, ry
        else:
            working_frame = frame
        
        # Determine if ball is moving fast (adjust thresholds accordingly)
        is_fast_motion = self.current_speed_estimate > 30  # pixels/frame
        
        # Adaptive thresholds based on motion
        if is_fast_motion:
            min_circularity = 0.3   # Very relaxed for motion blur
            min_solidity = 0.4
            min_aspect = 0.3        # Allow elongated shapes
            max_aspect = 3.5
        else:
            min_circularity = 0.5
            min_solidity = 0.6
            min_aspect = 0.5
            max_aspect = 2.0
        
        candidates = []
        
        # === METHOD 1: Color-based detection ===
        blurred = cv2.GaussianBlur(working_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        lower = self.hsv_ranges[self.ball_color]["lower"]
        upper = self.hsv_ranges[self.ball_color]["upper"]
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find color-based candidates
        color_candidates = self._find_candidates(
            color_mask, roi_offset_x, roi_offset_y,
            min_circularity, min_solidity, min_aspect, max_aspect,
            source="color"
        )
        candidates.extend(color_candidates)
        
        # === METHOD 2: Frame differencing (motion detection) ===
        if self.use_frame_differencing and self.prev_frame_gray is not None:
            current_gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
            
            # Compute difference
            frame_diff = cv2.absdiff(self.prev_frame_gray, current_gray)
            _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect nearby motion pixels
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
            
            # Combine with color mask (motion AND color = high confidence)
            combined_mask = cv2.bitwise_and(color_mask, motion_mask)
            
            # Also check pure motion in predicted area (for very fast/blurred ball)
            if self.last_known_position is not None and is_fast_motion:
                motion_candidates = self._find_candidates_in_search_area(
                    motion_mask, working_frame, roi_offset_x, roi_offset_y,
                    min_circularity=0.2, min_solidity=0.3,  # Very relaxed
                    source="motion"
                )
                candidates.extend(motion_candidates)
            
            # Update previous frame
            self.prev_frame_gray = current_gray
        else:
            # Initialize previous frame
            self.prev_frame_gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            self.prev_frame_gray = cv2.GaussianBlur(self.prev_frame_gray, (5, 5), 0)
        
        if not candidates:
            return self._predict_ball_position()
        
        # Score and select best candidate
        best = self._select_best_candidate(candidates)
        
        if best is None:
            return self._predict_ball_position()
        
        x, y, w, h = best['bbox']
        cx, cy = best['center']
        
        # Update tracking state
        if self.last_known_position is not None:
            dx = cx - self.last_known_position[0]
            dy = cy - self.last_known_position[1]
            self.current_speed_estimate = np.sqrt(dx**2 + dy**2)
            self.velocity = (dx, dy)
        
        self.last_known_position = (cx, cy)
        self.frames_since_detection = 0
        self.recent_ball_sizes.append(best['area'])
        
        return (x, y, w, h, cx, cy)
    
    def _find_candidates(self, mask, roi_offset_x, roi_offset_y,
                         min_circularity, min_solidity, min_aspect, max_aspect,
                         source="color"):
        """Find ball candidates from a binary mask."""
        candidates = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_ball_area or area > self.max_ball_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h if h > 0 else 0
            if aspect < min_aspect or aspect > max_aspect:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # For fast motion, check if elongation aligns with velocity direction
            motion_aligned = False
            if self.velocity != (0, 0) and aspect > 1.5:
                vel_angle = np.arctan2(self.velocity[1], self.velocity[0])
                # Get contour orientation
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    contour_angle = np.radians(ellipse[2])
                    angle_diff = abs(vel_angle - contour_angle)
                    angle_diff = min(angle_diff, np.pi - angle_diff)
                    motion_aligned = angle_diff < np.pi / 4  # Within 45 degrees
            
            # Relaxed check for motion-aligned elongated shapes
            if circularity < min_circularity and not motion_aligned:
                continue
            if solidity < min_solidity:
                continue
            
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx) + roi_offset_x, int(cy) + roi_offset_y
            
            candidates.append({
                'bbox': (x + roi_offset_x, y + roi_offset_y, w, h),
                'center': (cx, cy),
                'area': area,
                'circularity': circularity,
                'solidity': solidity,
                'source': source,
                'motion_aligned': motion_aligned
            })
        
        return candidates
    
    def _find_candidates_in_search_area(self, mask, frame, roi_offset_x, roi_offset_y,
                                         min_circularity, min_solidity, source="motion"):
        """Find candidates in a search window around predicted position."""
        candidates = []
        
        if self.last_known_position is None:
            return candidates
        
        # Predict where ball should be
        pred_x = self.last_known_position[0] + self.velocity[0] * 2
        pred_y = self.last_known_position[1] + self.velocity[1] * 2
        
        # Define search window
        search_radius = int(self.max_ball_speed_pixels)
        x1 = max(0, int(pred_x - roi_offset_x - search_radius))
        y1 = max(0, int(pred_y - roi_offset_y - search_radius))
        x2 = min(mask.shape[1], int(pred_x - roi_offset_x + search_radius))
        y2 = min(mask.shape[0], int(pred_y - roi_offset_y + search_radius))
        
        if x2 <= x1 or y2 <= y1:
            return candidates
        
        search_mask = mask[y1:y2, x1:x2]
        
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_ball_area * 0.5 or area > self.max_ball_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Very relaxed for motion-based detection in search area
            if circularity < min_circularity or solidity < min_solidity:
                continue
            
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            abs_x = int(cx) + x1 + roi_offset_x
            abs_y = int(cy) + y1 + roi_offset_y
            
            candidates.append({
                'bbox': (x + x1 + roi_offset_x, y + y1 + roi_offset_y, w, h),
                'center': (abs_x, abs_y),
                'area': area,
                'circularity': circularity,
                'solidity': solidity,
                'source': source,
                'motion_aligned': True  # Assume aligned since in search area
            })
        
        return candidates
    
    def _select_best_candidate(self, candidates):
        """Score and select the best ball candidate."""
        if not candidates:
            return None
        
        scored = []
        for c in candidates:
            # Shape score
            shape_score = c['circularity'] * 0.5 + c['solidity'] * 0.5
            
            # Proximity score
            proximity_score = 1.0
            if self.last_known_position is not None:
                cx, cy = c['center']
                last_x, last_y = self.last_known_position
                
                # Predict where ball should be
                pred_x = last_x + self.velocity[0]
                pred_y = last_y + self.velocity[1]
                
                # Distance to predicted position (better than distance to last position)
                dist_to_pred = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                dist_to_last = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                
                # Use the smaller distance
                distance = min(dist_to_pred, dist_to_last)
                
                if distance < self.max_ball_speed_pixels:
                    proximity_score = 1.0 - (distance / self.max_ball_speed_pixels) * 0.5
                else:
                    proximity_score = 0.2
            
            # Size consistency score
            size_score = 1.0
            if len(self.recent_ball_sizes) >= 3:
                avg_size = np.mean(self.recent_ball_sizes)
                size_diff = abs(c['area'] - avg_size) / avg_size if avg_size > 0 else 0
                size_score = max(0.3, 1.0 - size_diff * 0.5)
            
            # Bonus for motion-aligned detections during fast motion
            motion_bonus = 1.2 if c.get('motion_aligned', False) else 1.0
            
            # Source bonus (color detection is generally more reliable)
            source_bonus = 1.0 if c['source'] == 'color' else 0.8
            
            total = (shape_score * 0.25 + proximity_score * 0.45 + size_score * 0.30) * motion_bonus * source_bonus
            
            scored.append((total, c))
        
        # Return highest scoring candidate
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None
    
    def _predict_ball_position(self):
        """
        Predict ball position when detection fails, using velocity estimation.
        Uses momentum-based prediction for fast balls.
        Returns predicted bbox or None if prediction not possible.
        """
        self.frames_since_detection += 1
        
        # Allow more prediction frames for fast-moving balls
        max_frames = self.max_prediction_frames
        if self.current_speed_estimate > 50:
            max_frames = int(self.max_prediction_frames * 1.5)
        
        if self.frames_since_detection > max_frames:
            # Lost tracking - reset
            self.last_known_position = None
            self.velocity = (0, 0)
            self.current_speed_estimate = 0
            return None
        
        if self.last_known_position is None:
            return None
        
        # For very fast motion with no velocity, can't predict
        if self.velocity == (0, 0):
            return None
        
        # Predict new position based on velocity
        # Apply slight deceleration for more realistic prediction
        decay = 0.95 ** self.frames_since_detection
        pred_x = int(self.last_known_position[0] + self.velocity[0] * decay)
        pred_y = int(self.last_known_position[1] + self.velocity[1] * decay)
        
        # Update last known position with prediction
        self.last_known_position = (pred_x, pred_y)
        
        # Return estimated bbox using average recent size
        if self.recent_ball_sizes:
            avg_area = np.mean(self.recent_ball_sizes)
            estimated_size = int(np.sqrt(avg_area))
        else:
            estimated_size = 25
        
        # For fast motion, make bbox slightly larger (motion blur)
        if self.current_speed_estimate > 30:
            estimated_size = int(estimated_size * 1.3)
        
        half_size = estimated_size // 2
        
        return (pred_x - half_size, pred_y - half_size, 
                estimated_size, estimated_size, pred_x, pred_y)
    
    def calculate_speed(self, current_pos, current_time):
        """
        Calculate speed based on position change over time.
        Returns speed in m/s and km/h
        """
        if not self.calibrated:
            return 0, 0
        
        self.positions.append(current_pos)
        self.timestamps.append(current_time)
        
        if len(self.positions) < 2:
            return 0, 0
        
        # Calculate speed using recent positions (smoothing)
        # Use positions from a few frames back for stability
        lookback = min(5, len(self.positions) - 1)
        
        pos1 = self.positions[-lookback-1]
        pos2 = self.positions[-1]
        time1 = self.timestamps[-lookback-1]
        time2 = self.timestamps[-1]
        
        # Distance in pixels
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        distance_meters = distance_pixels / self.pixels_per_meter
        
        # Time difference
        dt = time2 - time1
        
        if dt <= 0:
            return self.current_speed_mps, self.current_speed_kmh
        
        # Speed in m/s
        speed_mps = distance_meters / dt
        
        # Apply smoothing
        self.speed_history.append(speed_mps)
        smoothed_speed_mps = np.mean(self.speed_history)
        
        # Convert to km/h
        speed_kmh = smoothed_speed_mps * 3.6
        
        self.current_speed_mps = smoothed_speed_mps
        self.current_speed_kmh = speed_kmh
        
        return smoothed_speed_mps, speed_kmh
    
    def draw_tracking(self, frame, bbox, speed_kmh, is_predicted=False):
        """
        Draw bounding box, speed, and optional tracking trail on frame.
        """
        if bbox is None:
            # Show message if ball not detected
            cv2.putText(frame, "Ball not detected", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        x, y, w, h, cx, cy = bbox
        
        # Add to trail if enabled
        if self.show_trail:
            self.trail_points.append((cx, cy))
            # Draw trail
            for i in range(1, len(self.trail_points)):
                if self.trail_points[i-1] is None or self.trail_points[i] is None:
                    continue
                thickness = max(1, int(np.sqrt(self.buffer_size / float(i + 1)) * 2))
                cv2.line(frame, self.trail_points[i-1], self.trail_points[i], (0, 165, 255), thickness)
        
        # Choose color based on whether this is detected or predicted
        if is_predicted:
            box_color = (0, 165, 255)  # Orange for predicted
            label_prefix = "[PRED] "
        else:
            box_color = (0, 255, 0)    # Green for detected
            label_prefix = ""
        
        # Draw bounding box
        padding = 5
        cv2.rectangle(frame, (x - padding, y - padding), 
                     (x + w + padding, y + h + padding), box_color, 2)
        
        # Draw center point
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        
        # Draw speed above bounding box
        speed_text = f"{label_prefix}{speed_kmh:.1f} km/h"
        text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text (ensure it stays on screen)
        text_x = max(5, x - padding)
        text_y = max(text_size[1] + 15, y - padding - 8)
        
        cv2.rectangle(frame, 
                     (text_x - 3, text_y - text_size[1] - 3),
                     (text_x + text_size[0] + 3, text_y + 3),
                     (0, 0, 0), -1)
        
        # Speed text
        cv2.putText(frame, speed_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def draw_info_overlay(self, frame, fps, frame_num, total_frames, output_fps=None):
        """
        Draw information overlay on the frame.
        """
        h, w = frame.shape[:2]
        
        # Calculate panel height based on content
        num_lines = 5
        if output_fps and output_fps != fps:
            num_lines = 6
        panel_height = 25 + num_lines * 22
        
        # Semi-transparent background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Info text
        info_lines = [
            f"Frame: {frame_num}/{total_frames}",
            f"Original FPS: {fps:.1f}",
            f"Speed: {self.current_speed_kmh:.1f} km/h ({self.current_speed_mps:.1f} m/s)",
            f"Ball Color: {self.ball_color}"
        ]
        
        if self.calibrated:
            info_lines.append(f"Scale: {self.pixels_per_meter:.1f} px/m")
        
        # Add slow motion indicator
        if output_fps and output_fps != fps:
            slowdown = fps / output_fps
            info_lines.append(f"Slow-Mo: {slowdown:.1f}x ({output_fps:.1f} FPS)")
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # Draw table outline if calibrated
        if self.calibrated and len(self.table_corners) == 4:
            pts = np.array(self.table_corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        
        return frame
    
    def process_video(self, show_preview=True, tune_detection=False):
        """
        Main processing loop - reads video, tracks ball, outputs annotated video.
        """
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds\n")
        
        # Read first frame for calibration
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return False
        
        # Calibration
        if not self.calibrate_from_table(first_frame):
            print("Calibration cancelled. Using default scale (may be inaccurate)")
            self.pixels_per_meter = 500  # Default fallback
            self.calibrated = True
        
        # Optional: tune ball detection
        if tune_detection:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, tune_frame = cap.read()
            if ret:
                self.tune_ball_detection(tune_frame)
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate output FPS for slow motion
        output_fps = fps  # Default: same as input
        
        if self.slow_mo_fps is not None:
            output_fps = self.slow_mo_fps
            print(f"Slow motion: Output at {output_fps} FPS (original: {fps} FPS)")
        elif self.slow_mo_factor is not None:
            output_fps = fps / self.slow_mo_factor
            print(f"Slow motion: {self.slow_mo_factor}x slower ({output_fps:.1f} FPS, original: {fps} FPS)")
        
        # Setup video writer with output FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, output_fps, (width, height))
        
        print(f"Processing video...")
        print(f"Output will be saved to: {self.output_path}")
        if output_fps != fps:
            slowdown = fps / output_fps
            print(f"Playback will be {slowdown:.1f}x slower than real-time\n")
        else:
            print()
        
        frame_num = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            current_time = frame_num / fps
            
            # Detect ball
            bbox = self.detect_ball(frame)
            
            # Check if this is a predicted position
            is_predicted = self.frames_since_detection > 0
            
            # Calculate speed
            if bbox is not None:
                center = (bbox[4], bbox[5])
                speed_mps, speed_kmh = self.calculate_speed(center, current_time)
            else:
                speed_mps, speed_kmh = self.current_speed_mps, self.current_speed_kmh
            
            # Draw tracking visualization
            frame = self.draw_tracking(frame, bbox, speed_kmh, is_predicted)
            frame = self.draw_info_overlay(frame, fps, frame_num, total_frames, output_fps)
            
            # Write frame to output
            out.write(frame)
            
            # Show preview
            if show_preview:
                cv2.imshow("Ball Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nProcessing cancelled by user")
                    break
                elif key == ord('p'):
                    # Pause
                    cv2.waitKey(0)
            
            # Progress update
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_num) * (total_frames - frame_num)
                print(f"Progress: {progress:.1f}% | Frame {frame_num}/{total_frames} | ETA: {eta:.1f}s", end='\r')
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n\n✓ Processing complete!")
        print(f"  Processed {frame_num} frames in {elapsed:.1f} seconds")
        print(f"  Output saved to: {self.output_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Table Tennis Ball Tracker - Track ball and calculate speed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ball_tracker.py video.mp4
  python ball_tracker.py video.mp4 -o tracked_output.mp4
  python ball_tracker.py video.mp4 --ball-color white --tune
  python ball_tracker.py video.mp4 --no-preview
        """
    )
    
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output_tracked.mp4",
                       help="Path for output video (default: output_tracked.mp4)")
    parser.add_argument("--ball-color", choices=["orange", "white", "yellow"],
                       default="orange", help="Ball color for detection (default: orange)")
    parser.add_argument("--tune", action="store_true",
                       help="Open tuning window to adjust ball detection parameters")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable live preview (faster processing)")
    parser.add_argument("--trail", action="store_true",
                       help="Show ball trajectory trail (orange line)")
    parser.add_argument("--min-area", type=int, default=150,
                       help="Minimum ball area in pixels (default: 150)")
    parser.add_argument("--max-area", type=int, default=8000,
                       help="Maximum ball area in pixels (default: 8000)")
    parser.add_argument("--slow-mo", type=float, default=None,
                       help="Save output in slow motion at specified FPS (e.g., --slow-mo 5 for 5 FPS)")
    parser.add_argument("--slow-mo-factor", type=float, default=None,
                       help="Slow down by factor (e.g., --slow-mo-factor 4 = 4x slower)")
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = TableTennisTracker(args.video, args.output)
    tracker.set_ball_color(args.ball_color)
    tracker.show_trail = args.trail
    tracker.min_ball_area = args.min_area
    tracker.max_ball_area = args.max_area
    
    # Set slow motion options
    tracker.slow_mo_fps = args.slow_mo
    tracker.slow_mo_factor = args.slow_mo_factor
    
    # Process video
    tracker.process_video(
        show_preview=not args.no_preview,
        tune_detection=args.tune
    )


if __name__ == "__main__":
    main()