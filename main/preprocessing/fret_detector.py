# guitar_neck_detector.py
import cv2
import numpy as np
import yaml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict
from .fret_geometry import FretboardValidator

class GuitarNeckDetector:
    """
    Detects and validates a guitar fretboard in real-time with production-grade features.
    Version: 6.0 (Customer-Ready)
    """
    def __init__(self, config_path: str, verbose: bool = False):
        self.params = self._load_config(config_path)
        self.verbose = verbose
        # --- PRODUCTION POLISH: State for tracking and warnings ---
        self.frets: List[np.ndarray] = []
        self.strings: List[np.ndarray] = []
        self.last_known_frets: List[np.ndarray] = []
        self.last_known_strings: List[np.ndarray] = []
        self.detection_confidence: float = 0.0
        self.warnings: List[str] = []

    def _load_config(self, path: str) -> Dict:
        """Loads parameters from a YAML file with validation."""
        default_config = {"hough_thresh": 30, "hough_min_len_ratio": 0.12, "dbscan_eps": 10}
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                # --- PRODUCTION POLISH: Validate loaded config ---
                if not isinstance(config, dict):
                    raise ValueError("Malformed YAML config: root is not a dictionary.")
                return config
        except FileNotFoundError:
            self._log(f"Config file not found at {path}. Using default parameters.")
            return default_config
        except Exception as e:
            self._log(f"Error loading config: {e}. Using default parameters.")
            return default_config

    def _log(self, message: str):
        if self.verbose: print(f"[Detector] {message}")
    
    def _sort_lines_by_pca(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """Sorts lines along their primary axis using PCA, robust to rotation."""
        if len(lines) < 2: return lines
        midpoints = np.array([[(l[0] + l[2])/2, (l[1] + l[3])/2] for l in lines])
        pca = PCA(n_components=1).fit(midpoints)
        projections = midpoints @ pca.components_.T
        return [lines[i] for i in np.argsort(projections[:, 0])]

    def detect_fretboard(self, frame: np.ndarray) -> bool:
        """Performs a full detection and geometric validation of the fretboard."""
        self.warnings.clear()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        min_line_length = int(frame.shape[1] * self.params["hough_min_len_ratio"])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.params["hough_thresh"], minLineLength=min_line_length, maxLineGap=20)

        if lines is None or len(lines) < 4:
            self.warnings.append("Insufficient raw lines detected.")
            self.detection_confidence = 0.0
            return False

        angles = np.array([np.rad2deg(np.arctan2(y2-y1, x2-x1)) % 180 for x1,y1,x2,y2 in lines[:,0]])
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(angles.reshape(-1,1))
        centers = kmeans.cluster_centers_.flatten()
        
        # --- PRODUCTION POLISH: More direct and robust label assignment ---
        fret_label = np.argmin([abs(c - 90) for c in centers])
        
        def merge(line_group):
            if len(line_group) < 2: return []
            rhos = [(l[0][0]*l[0][3]-l[0][2]*l[0][1])/np.sqrt((l[0][2]-l[0][0])**2+(l[0][3]-l[0][1])**2) for l in line_group]
            db = DBSCAN(eps=self.params["dbscan_eps"], min_samples=2).fit(np.array(rhos).reshape(-1,1))
            # --- PRODUCTION POLISH: Sanitize structure immediately after merging ---
            return [np.mean(line_group[db.labels_==lbl], axis=0, dtype=np.int32)[0] for lbl in set(db.labels_) if lbl!=-1]

        frets = merge(lines[kmeans.labels_ == fret_label])
        strings = merge(lines[kmeans.labels_ != fret_label])
        
        fret_score = FretboardValidator.validate_fret_spacing(frets)
        string_score = FretboardValidator.validate_string_spacing(strings)
        if fret_score < 0.6: self.warnings.append("Fret spacing inconsistent with 12-TET rule.")
        if string_score < 0.6: self.warnings.append("String spacing inconsistent.")

        confidence_angle = np.clip(1.0 - abs(abs(centers[0]-centers[1]) - 90) / 45, 0, 1)
        self.detection_confidence = (confidence_angle*0.4) + (fret_score*0.3) + (string_score*0.3)
        self._log(f"Final confidence: {self.detection_confidence:.2f} | Warnings: {self.warnings}")
        
        if self.is_valid():
            self.frets = self._sort_lines_by_pca(frets)
            self.strings = self._sort_lines_by_pca(strings)
            self.last_known_frets, self.last_known_strings = self.frets, self.strings
            return True
        return False

    def track_fretboard(self, frame: np.ndarray, frame_count: int) -> bool:
        """--- PRODUCTION POLISH: Real-time tracking method ---"""
        if frame_count % 30 == 0 or self.detection_confidence < 0.3:
            return self.detect_fretboard(frame)
        else:
            self.frets = self.last_known_frets
            self.strings = self.last_known_strings
            return len(self.frets) > 0

    def get_grid_cell_details(self, fingertip_coord: Tuple[int, int]) -> Optional[Dict]:
        """Snaps a fingertip to the grid using PCA-sorted lines."""
        if not self.is_valid(): return None
        pt = np.array(fingertip_coord)
        def get_dist(p, line):
            p1, p2 = np.array([line[0], line[1]]), np.array([line[2], line[3]])
            norm = np.linalg.norm(p2 - p1)
            return np.abs(np.cross(p2 - p1, p1 - p)) / norm if norm != 0 else float('inf')
        
        string_dists = [get_dist(pt, s) for s in self.strings]
        fret_dists = [get_dist(pt, f) for f in self.frets]
        return {
            "string_idx": np.argmin(string_dists), "fret_idx": np.argmin(fret_dists),
            "string_dist_px": np.min(string_dists), "fret_dist_px": np.min(fret_dists)
        }

    @property
    def is_valid(self) -> bool:
        """--- PRODUCTION POLISH: Test hook for validity ---"""
        return self.detection_confidence > 0.6 and not self.warnings

    def get_report(self) -> Dict:
        """--- PRODUCTION POLISH: Diagnostic report for GUI/dev tools ---"""
        return {
            "is_valid": self.is_valid,
            "frets_detected": len(self.frets),
            "strings_detected": len(self.strings),
            "confidence": self.detection_confidence,
            "warnings": self.warnings
        }

    def draw_detected_grid(self, frame: np.ndarray) -> np.ndarray:
        """--- PRODUCTION POLISH: Enhanced visualization ---"""
        vis_frame = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.frets):
            color = (0, 255 - i*15, i*15) # Color fades from green to blue
            cv2.line(vis_frame, (x1, y1), (x2, y2), color, 2)
        for i, (x1, y1, x2, y2) in enumerate(self.strings):
            color = (255, i*20, 0) # Color fades from blue to red
            cv2.line(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        report = self.get_report()
        text = f"Confidence: {report['confidence']:.2f}"
        color = (0, 255, 0) if report['is_valid'] else (0, 0, 255)
        cv2.putText(vis_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return vis_frame