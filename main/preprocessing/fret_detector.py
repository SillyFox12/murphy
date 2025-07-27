import cv2
import numpy as np
import yaml
import requests
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict
from fret_geometry import FretboardValidator
from fret_debug import FretboardDebugVisualizer

class GuitarNeckDetector:
    """
    Detects and validates a guitar fretboard and stores intermediate data for debugging.
    Version: 12.0 (Debug Edition)
    """
    def __init__(self, config_path: str, verbose: bool = False):
        self.params = self._load_config(config_path)
        self.verbose = verbose
        # State and Debugging
        self.frets: List[np.ndarray] = []
        self.strings: List[np.ndarray] = []
        self.detection_confidence: float = 0.0
        self.warnings: List[str] = []
        self.debug_data: Dict = {}

    def _load_config(self, path: str) -> Dict:
        default_config = {"hough_thresh": 50, "hough_min_len_ratio": 0.3, "hough_max_gap": 50, 
                          "dbscan_min_samples": 2, 
                          "min_fret_intersections": 2, "min_string_intersections": 3, 
                          "valid_confidence_threshold": 0.3}
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f); return {**default_config, **config}
        except Exception:
            self._log(f"Config file not found or invalid. Using defaults.")
            return default_config
            
    def _log(self, message: str):
        if self.verbose: print(f"[Detector] {message}")

    def _get_hough_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        blur = cv2.bilateralFilter(enhanced_gray, 5, 75, 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
        edges = cv2.Canny(tophat, 100, 180)

        self.debug_data['preprocessed'] = blur
        self.debug_data['edges'] = edges
        
        min_line_length = int(frame.shape[1] * self.params["hough_min_len_ratio"])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.params["hough_thresh"], minLineLength=min_line_length, maxLineGap=self.params["hough_max_gap"])
        self._log(f"Found {len(lines) if lines is not None else 0} raw line segments.")
        return lines

    def _estimate_dbscan_eps(self, data: np.ndarray, min_samples: int) -> float:
        if len(data) < min_samples: return 10.0
        nn = NearestNeighbors(n_neighbors=min_samples).fit(data)
        distances, _ = nn.kneighbors(data)
        k_distances = np.sort(distances[:, -1])
        return max(np.percentile(k_distances, 95), 5.0)

    def _filter_and_merge_lines(self, lines: np.ndarray) -> List[np.ndarray]:
        min_samples = self.params["dbscan_min_samples"]
        if len(lines) < min_samples: return []
        rhos = np.array([(l[0][0]*l[0][3] - l[0][2]*l[0][1]) / np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines]).reshape(-1, 1)
        estimated_eps = self._estimate_dbscan_eps(rhos, min_samples)
        self._log(f"Using auto-estimated DBSCAN eps: {estimated_eps:.2f}")
        db = DBSCAN(eps=estimated_eps, min_samples=min_samples).fit(rhos)
        
        merged_lines = []
        for label in set(db.labels_):
            if label == -1: continue
            merged_lines.append(np.mean(lines[db.labels_ == label], axis=0, dtype=np.int32)[0])
        return merged_lines

    def detect_fretboard(self, frame: np.ndarray) -> bool:
        self.warnings.clear()
        self.debug_data = {'original': frame.copy()}
        lines = self._get_hough_lines(frame)
        self.debug_data['raw_lines'] = lines

        if lines is None or len(lines) < 10:
            self.warnings.append("Insufficient raw lines detected."); self.detection_confidence = 0.0; return False

        angles = np.array([np.rad2deg(np.arctan2(y2-y1, x2-x1)) % 180 for x1,y1,x2,y2 in lines[:,0]])
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(angles.reshape(-1,1))
        centers = kmeans.cluster_centers_.flatten()
        fret_label = np.argmin([abs(c - 90) for c in centers])
        
        raw_frets = lines[kmeans.labels_ == fret_label]
        raw_strings = lines[kmeans.labels_ != fret_label]
        
        fret_scores = [0] * len(raw_frets); string_scores = [0] * len(raw_strings)
        for i, str_line in enumerate(raw_strings):
            for j, frt_line in enumerate(raw_frets):
                x1, y1, x2, y2 = str_line[0]; x3, y3, x4, y4 = frt_line[0]
                den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if den == 0: continue
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
                if 0 <= t <= 1 and 0 <= u <= 1:
                    string_scores[i] += 1; fret_scores[j] += 1

        grid_frets = np.array([raw_frets[i] for i, s in enumerate(fret_scores) if s >= self.params["min_fret_intersections"]])
        grid_strings = np.array([raw_strings[i] for i, s in enumerate(string_scores) if s >= self.params["min_string_intersections"]])
        self.debug_data['grid_filtered_frets'] = grid_frets
        self.debug_data['grid_filtered_strings'] = grid_strings
        self._log(f"Grid intersection filter kept {len(grid_frets)} frets and {len(grid_strings)} strings.")

        if len(grid_frets) < 3 or len(grid_strings) < 3:
            self.warnings.append("Could not find a cohesive grid structure."); self.detection_confidence = 0.0; return False

        frets = self._filter_and_merge_lines(grid_frets)
        strings = self._filter_and_merge_lines(grid_strings)
        self.debug_data['merged_frets'] = frets
        self.debug_data['merged_strings'] = strings

        fret_score = FretboardValidator.validate_fret_spacing(frets)
        string_score = FretboardValidator.validate_string_spacing(strings)
        if fret_score < 0.6: self.warnings.append("Fret spacing inconsistent.")
        if string_score < 0.6: self.warnings.append("String spacing inconsistent.")
        
        confidence_angle = np.clip(1.0 - abs(abs(centers[0]-centers[1]) - 90) / 45, 0, 1)
        self.detection_confidence = (confidence_angle*0.4) + (fret_score*0.3) + (string_score*0.3)
        self._log(f"Final confidence: {self.detection_confidence:.2f} | Warnings: {self.warnings}")
        
        if self.is_valid:
            self.frets = self._sort_lines_by_pca(frets)
            self.strings = self._sort_lines_by_pca(strings)
            return True
        return False

    @property
    def is_valid(self) -> bool:
        return self.detection_confidence > self.params["valid_confidence_threshold"] and not self.warnings

    def _sort_lines_by_pca(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        if len(lines) < 2: return lines
        midpoints = np.array([[(l[0] + l[2])/2, (l[1] + l[3])/2] for l in lines])
        pca = PCA(n_components=1).fit(midpoints)
        projections = midpoints @ pca.components_.T
        return [lines[i] for i in np.argsort(projections[:, 0])]

# ==============================================================================
# DEMONSTRATION BLOCK
# ==============================================================================
if __name__ == '__main__':
    CONFIG_FILE_PATH = "detector_config.yml"
    IMAGE_URL = "https://www.unlv.edu/sites/default/files/styles/1200_width/public/releases/main-images/Ana%20Vidovic_main.jpg?itok=I69lZLtY"
    
    with open(CONFIG_FILE_PATH, "w") as f: yaml.dump({}, f)

    try:
        response = requests.get(IMAGE_URL, stream=True, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.raw.read()), dtype="uint8")
        sample_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        print("✅ Successfully loaded sample image.")
    except Exception as e:
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = GuitarNeckDetector(config_path=CONFIG_FILE_PATH, verbose=True)
    detector.detect_fretboard(sample_frame)
    
    print("\n--- Generating Debug Panel ---")
    visualizer = FretboardDebugVisualizer(detector)
    visualizer.generate_debug_panel()