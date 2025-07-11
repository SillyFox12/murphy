# -*- coding: utf-8 -*-
import csv
import os
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional

class HandPoseFeatureEngineer:
    """Calculates a feature vector of geometric properties from hand landmarks."""
    # The landmark indices for each finger from thumb to pinky.
    # Each tuple contains (Tip, DIP, PIP, MCP) joints.
    FINGER_LANDMARK_INDICES = {
        "THUMB": (4, 3, 2, 1), 
        "INDEX": (8, 7, 6, 5), 
        "MIDDLE": (12, 11, 10, 9), 
        "RING": (16, 15, 14, 13), 
        "PINKY": (20, 19, 18, 17)
    }

    def __init__(self):
        pass

    @staticmethod
    def _calculate_distance(p1, p2) -> float:
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    @staticmethod
    def _calculate_angle(p1, p2, p3) -> float:
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def get_feature_names(self) -> list[str]:
        """Generates the list of feature names for the CSV header."""
        # This is a robust way to ensure the header always matches the features.
        names = []
        for finger in self.FINGER_LANDMARK_INDICES.keys():
            names.extend([f'angle_{finger.lower()}_pip', f'angle_{finger.lower()}_dip'])
        
        finger_names = list(self.FINGER_LANDMARK_INDICES.keys())
        for i in range(len(finger_names)):
            for j in range(i + 1, len(finger_names)):
                names.append(f'dist_{finger_names[i].lower()}_{finger_names[j].lower()}')

        names.extend(['thumb_y_rel_wrist', 'thumb_y_rel_index_mcp', 'thumb_dist_to_index_mcp'])
        return names

    def calculate_features(self, hand_landmarks) -> Optional[Dict[str, float]]:
        if not hand_landmarks: return None
        lm = hand_landmarks.landmark
        features = {}
        for finger, indices in self.FINGER_LANDMARK_INDICES.items():
            tip, dip, pip, mcp = indices
            features[f'angle_{finger.lower()}_pip'] = self._calculate_angle(lm[mcp], lm[pip], lm[dip])
            features[f'angle_{finger.lower()}_dip'] = self._calculate_angle(lm[pip], lm[dip], lm[tip])
        
        fingertip_indices = [indices[0] for indices in self.FINGER_LANDMARK_INDICES.values()]
        finger_names = list(self.FINGER_LANDMARK_INDICES.keys())
        for i in range(len(fingertip_indices)):
            for j in range(i + 1, len(fingertip_indices)):
                p1_idx, p2_idx = fingertip_indices[i], fingertip_indices[j]
                features[f'dist_{finger_names[i].lower()}_{finger_names[j].lower()}'] = self._calculate_distance(lm[p1_idx], lm[p2_idx])

        features['thumb_y_rel_wrist'] = lm[4].y - lm[0].y
        features['thumb_y_rel_index_mcp'] = lm[4].y - lm[5].y
        features['thumb_dist_to_index_mcp'] = self._calculate_distance(lm[4], lm[5])
        return features


# ==============================================================================
# SCRIPT 2: The New, Intelligent Feature Logger ⚙️
# ==============================================================================

class FeatureLogger:
    """
    An intelligent logger that calculates and saves engineered features to a CSV file.
    """
    def __init__(self, output_dir: str, filename: str = "hand_features.csv"):
        self.output_path = os.path.join(output_dir, filename)
        self.feature_engineer = HandPoseFeatureEngineer()
        self.header = ["frame", "hand_index"] + self.feature_engineer.get_feature_names()
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        """Writes the CSV header if the file doesn't exist."""
        if not os.path.exists(self.output_path):
            with open(self.output_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)

    def log_features(self, frame_number: int, results):
        """
        Calculates features for detected hands and logs them to the CSV file.
        
        Args:
            frame_number: The current video frame number.
            results: The MediaPipe results object containing hand landmarks.
        """
        if not results.multi_hand_landmarks:
            return  # Skip frames where no hands are detected

        # Use 'a' (append) mode to add new rows without overwriting the file
        with open(self.output_path, mode='a', newline='', encoding='utf-8') as file:
            # DictWriter is robust for writing feature dictionaries
            writer = csv.DictWriter(file, fieldnames=self.header)

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Calculate the feature vector for the current hand
                feature_vector = self.feature_engineer.calculate_features(hand_landmarks)
                
                if feature_vector:
                    # Prepare the full row for the CSV
                    row_data = {"frame": frame_number, "hand_index": hand_idx}
                    row_data.update(feature_vector) # Merge the two dictionaries
                    writer.writerow(row_data)

# ==============================================================================
# SCRIPT 3: The Hand Tracker (Unchanged)
# ==============================================================================

class HandTracker:
        #Initializes mediapipe tools
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

    def process_frame(self, frame):
        #Flips the image horizontally
        frame = cv2.flip(frame, 1)  # 1 = horizontal flip

        #Converts BRG to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Creates an object with hand landmarks and handedness
        self.results = self.hands.process(rgb)
        return frame, self.results

    def display_frame(self, frame, results):
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        
    #Closes the viewing window
    def cleanup(self):
        cv2.destroyAllWindows()

# ==============================================================================
# DEMONSTRATION: Simulating the Main Pipeline Loop
# ==============================================================================

if __name__ == '__main__':
    # --- Setup ---
    # Create a dummy video frame (e.g., a black image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
    
    # Create an output directory
    output_directory = "./data"
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Initializing components. Output will be saved in '{output_directory}'.")
    tracker = HandTracker()
    # Use the NEW FeatureLogger instead of the old LandmarkLogger
    logger = FeatureLogger(output_dir=output_directory, filename="hand_features.csv")
    print(f"Logger initialized. Feature data will be saved to: {logger.output_path}")

    # --- Main Loop Simulation ---
    # This loop simulates what happens inside your `_run_video_analysis_process`
    print("\nSimulating video processing for 5 frames...")
    for frame_number in range(5):
        # 1. Process the frame to get landmarks
        frame, results = tracker.process_frame(dummy_frame)

        # In a real scenario, MediaPipe would find hands here.
        # Since our dummy frame is black, we'll manually add mock landmarks
        # to demonstrate that the logger is working.
        if results.multi_hand_landmarks is None:
             # A hack to ensure the results object has a list to append to
            from mediapipe.framework.formats import landmark_pb2
            mock_landmarks = landmark_pb2.NormalizedLandmarkList()
            for _ in range(21):
                mock_landmarks.landmark.add(x=np.random.rand(), y=np.random.rand(), z=np.random.rand())
            results.multi_hand_landmarks = [mock_landmarks]

        # 2. Log the CALCULATED FEATURES (not the raw landmarks)
        logger.log_features(frame_number, results)
        print(f"  - Processed frame {frame_number} and logged features.")

    tracker.cleanup()
    print("\n✅ Simulation complete. Check the 'hand_features.csv' file.")