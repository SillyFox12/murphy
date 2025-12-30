# -*- coding: utf-8 -*-
import csv
import os
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm

# ==============================================================================
# SCRIPT 1: The Corrected HandPoseFeatureEngineer
# ==============================================================================

class HandPoseFeatureEngineer:
    """Calculates a feature vector of geometric properties from hand landmarks."""
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
        """Calculates the Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    # Calculate the angle between three landmarks
    # p1 is the tip, p2 is the middle joint, p3 is the base joint
    @staticmethod
    def _calculate_angle(p1, p2, p3) -> float:
        """Calculates the angle in degrees between three landmarks. 
        p1 is the tip, p2 is the middle joint, p3 is the base joint."""
        
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    @staticmethod
    def _get_palm_plane(lm_list) -> Tuple[np.ndarray, np.ndarray]:
        """Defines a plane for the palm using the wrist, index MCP, and pinky MCP."""
        p1 = np.array([lm_list[0].x, lm_list[0].y, lm_list[0].z])
        p2 = np.array([lm_list[5].x, lm_list[5].y, lm_list[5].z])
        p3 = np.array([lm_list[17].x, lm_list[17].y, lm_list[17].z])
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else 1
        return normal, p1

    def get_feature_names(self) -> List[str]:
        """Generates the list of feature names for the CSV header."""
        names = []
        for finger in self.FINGER_LANDMARK_INDICES.keys():
            names.extend([f'angle_{finger.lower()}_pip', f'angle_{finger.lower()}_dip'])
        
        finger_names = list(self.FINGER_LANDMARK_INDICES.keys())
        for i in range(len(finger_names)):
            for j in range(i + 1, len(finger_names)):
                names.append(f'dist_{finger_names[i].lower()}_{finger_names[j].lower()}')

        names.extend([
            'thumb_y_rel_wrist', 'thumb_y_rel_index_mcp', 'thumb_dist_to_index_mcp',
            # --- FIX: Added new feature names to the header ---
            'thumb_palm_dist', 'thumb_adduction_angle'
        ])
        return names

    def calculate_features(self, hand_landmarks: mp.solutions.hands.HandLandmark) -> Optional[Dict[str, float]]:
        """
        Calculates a feature vector from hand landmarks.
        Returns a dictionary of features or None if no landmarks are provided.
        """
        if not hand_landmarks:
            return None
            
        lm = hand_landmarks.landmark
        features = {}

        # 1. Finger Curl Angles
        for finger, indices in self.FINGER_LANDMARK_INDICES.items():
            tip, dip, pip, mcp = indices
            features[f'angle_{finger.lower()}_pip'] = self._calculate_angle(lm[mcp], lm[pip], lm[dip])
            features[f'angle_{finger.lower()}_dip'] = self._calculate_angle(lm[pip], lm[dip], lm[tip])
        
        # 2. Inter-Finger Distances
        fingertip_indices = [indices[0] for indices in self.FINGER_LANDMARK_INDICES.values()]
        finger_names = list(self.FINGER_LANDMARK_INDICES.keys())
        for i in range(len(fingertip_indices)):
            for j in range(i + 1, len(fingertip_indices)):
                p1_idx, p2_idx = fingertip_indices[i], fingertip_indices[j]
                features[f'dist_{finger_names[i].lower()}_{finger_names[j].lower()}'] = self._calculate_distance(lm[p1_idx], lm[p2_idx])

        # 3. Original Thumb Positional Features
        features['thumb_y_rel_wrist'] = lm[4].y - lm[0].y
        features['thumb_y_rel_index_mcp'] = lm[4].y - lm[5].y
        features['thumb_dist_to_index_mcp'] = self._calculate_distance(lm[4], lm[5])

        # 4. Thumb-to-Palm Distance
        normal, p1 = self._get_palm_plane(lm)
        thumb_tip_vec = np.array([lm[4].x, lm[4].y, lm[4].z])
        features['thumb_palm_dist'] = np.dot((thumb_tip_vec - p1), normal)

        # 5. Thumb Adduction Angle
        wrist_vec = np.array([lm[0].x, lm[0].y, lm[0].z])
        mcp_middle_vec = np.array([lm[9].x, lm[9].y, lm[9].z])
        mcp_thumb_vec = np.array([lm[1].x, lm[1].y, lm[1].z])
        
        hand_axis = mcp_middle_vec - wrist_vec
        thumb_axis = mcp_thumb_vec - wrist_vec
        
        cos_angle = np.dot(hand_axis, thumb_axis) / (np.linalg.norm(hand_axis) * np.linalg.norm(thumb_axis))
        features['thumb_adduction_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return features

# ==============================================================================
# SCRIPT 2: Feature Logger
# ==============================================================================

class FeatureLogger:
    """An intelligent logger that calculates and saves engineered features to a CSV file."""
    def __init__(self, output_dir: str, filename: str = "hand_features.csv"):
        self.output_path = os.path.join(output_dir, filename)
        self.feature_engineer = HandPoseFeatureEngineer()
        self.header = ["frame", "hand_index"] + self.feature_engineer.get_feature_names() + ["wrist_error", "wrist_error_type",
                                                                                          "thumb_error", "thumb_error_type",
                                                                                          "finger_error", "finger_error_type"]
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        """Writes the CSV header using DictWriter if the file doesn't exist."""
        if not os.path.exists(self.output_path):
            with open(self.output_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.header)
                writer.writeheader()

    def log_features(self, frame_number: int, results):
        """Calculates features for detected hands and logs them to the CSV file."""
        if not results.multi_hand_landmarks:
            return

        with open(self.output_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.header)
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                feature_vector = self.feature_engineer.calculate_features(hand_landmarks)
                if feature_vector:
                    row_data = {"frame": frame_number, 
                                "hand_index": hand_idx,
                                "wrist_error": 0.0,
                                "wrist_error_type": "none",
                                "thumb_error": 0.0,
                                "thumb_error_type": "none",
                                "finger_error": 0.0,
                                "finger_error_type": "none"}
                    row_data.update(feature_vector)
                    writer.writerow(row_data)

# ==============================================================================
# SCRIPT 3: The Hand Tracker
# ==============================================================================

class HandTracker:
    """A simplified wrapper for MediaPipe Hands."""
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def process_frame(self, frame):
        """Processes a single video frame to find hand landmarks."""
        frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror effect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results
    
    def extract_features_from_frames(
        self,
        frames_dir: str,
        output_dir: str,
        output_csv: str = "hand_features.csv"
    ):
        """
        Feeds a folder of frames through MediaPipe and logs features to CSV.
        """
        os.makedirs(output_dir, exist_ok=True)

        tracker = HandTracker()
        logger = FeatureLogger(output_dir=output_dir, filename=output_csv)

        # Sort frames to preserve time order
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"Processing {len(frame_files)} frames...")

        for frame_idx, frame_name in enumerate(tqdm(frame_files)):
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"⚠️ Skipping unreadable frame: {frame_name}")
                continue

            results = tracker.process_frame(frame)
            logger.log_features(frame_idx, results)

        tracker.cleanup()
        print(f"\n✅ Feature extraction complete.")
        print(f"📄 Output saved to: {logger.output_path}")


    def cleanup(self):
        """Releases MediaPipe resources."""
        self.hands.close()

# ==============================================================================
# DEMONSTRATION: Simulating the Main Pipeline Loop
# ==============================================================================

if __name__ == '__main__':
    tracker = HandTracker()

    tracker.extract_features_from_frames(
        frames_dir=r"data/hand_data/wrist/raised",
        output_dir=r"data/processed_hand_data/wrist/raised",
        output_csv="raised_wrist.csv"
    )