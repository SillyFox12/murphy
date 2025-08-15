# Basic packages for data manipulation
import os
import cv2
import numpy as np
import yaml
from scipy.signal import find_peaks
from typing import Tuple, List, Optional, Dict

class Strings_Frets:
    """Extracts strings and frets form cropped fretboard image provided from fretboard_isolator.py"""
    def __init__(self, config_path: Optional[str], verbose: bool = False):
        self.verbose = verbose
        self.params = self._load_config(config_path)       
        # State and Debugging
        self.roi_img: np.ndarray = np.array([])
        self.frets: List[np.ndarray] = []
        self.strings: List[np.ndarray] = []
        self.detection_confidence: float = 0.0
        self.warnings: List[str] = []
        self.debug_data: Dict = {}

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """ Loads configuration parameters from a YAML file.
        parameters:
        - path: The file path to the YAML configuration file.
        returns:
        - A dictionary containing the configuration parameters.
        """
        default_config = {
            # Preprocessing
            'blur_kernel_size': (5, 5),
            'canny_threshold1_flat': 0.66,
            'canny_threshold2_flat': 1.33,
            'canny_threshold1_bright': 1.0,
            'canny_threshold2_bright': 1.66,
            # Hough Transform
            'string_threshold': 50,
            'string_min_len_ratio': 0.4,
            'string_max_gap_ratio': 0.10,
            'fret_threshold': 10,
            'fret_min_len_ratio': 0.3,
            'fret_max_gap_ratio': 8,
            # Clustering
            'string_tolerance': 8,
            'fret_tolerance': 8,
            'fret_ratio_tolerance': 0.25,  # Tolerance for fret spacing ratio validation
            'fret_min_span_ratio': 0.40,
            # Image Filtering
        }
        if config_path and isinstance(config_path, (str, bytes, os.PathLike)):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                default_config.update(config)
            except Exception as e:
                self._log(f"Failed to load config file. Loading defaults. Error: {e}")
        else:
            self._log("No config path provided. Using default configuration.")
        return default_config
    
    # Message logger
    def _log(self, message: str):
        """Logs a message to the console or a log file."""
        if self.verbose: print(f"[DEBUG] {message}")

    def _merge_lines(self, positions: List[int], tol: int) -> List[int]:
        """Simple 1D clustering: sort positions and group values within tol."""
        if not positions:
            self._log("No positions to merge.")
            return []
        
        pos_sorted = sorted(positions)
        clusters = [[pos_sorted[0]]]

        # Merge close positions
        for p in pos_sorted[1:]:
            if abs(p - np.mean(clusters[-1])) <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [int(sum(c) / len(c)) for c in clusters]
    
    def _validate_fret_spacing(self, fret_positions: List[int]) -> bool:
        """Checks if fret spacing follows the guitar's 1/(2^(1/12)) ratio within tolerance."""
        if len(fret_positions) < 3:
            self._log("Not enough frets detected for ratio validation.")
            return False

        fret_positions = sorted(fret_positions)
        target_ratio = 1 / (2 ** (1/12))
        tol = self.params.get('fret_ratio_tolerance', 0.05)

        valid_ratios = []
        for i in range(len(fret_positions) - 2):
            d1 = fret_positions[i+1] - fret_positions[i]
            d2 = fret_positions[i+2] - fret_positions[i+1]
            if d1 == 0:  # avoid division by zero
                continue
            ratio = d2 / d1
            valid = abs(ratio - target_ratio) <= tol
            valid_ratios.append(valid)
            self._log(f"Fret {i}->{i+1}->{i+2}: gap1={d1}, gap2={d2}, ratio={ratio:.4f}, valid={valid}")

        # Decide if geometry is acceptable (e.g., at least 80% valid ratios)
        if valid_ratios and sum(valid_ratios) / len(valid_ratios) >= 0.8:
            self._log("Fret spacing passes geometric validation.")
            return True
        else:
            self._log("Fret spacing fails geometric validation.")
            return False
        
    def _image_filter(self, roi_bgr: np.ndarray) -> np.ndarray:

        # --- STRING PREPROCESS ---
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab)
        gray = lab[:, :, 0]
        self.debug_data['gray'] = gray

        se_radius = self.params.get('background_se_radius', 15)
        background = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_radius, se_radius))
        )
        flattened = cv2.subtract(gray, background)
        self.debug_data['background_flattened'] = flattened

        clahe = cv2.createCLAHE(
            clipLimit=self.params.get('clahe_clip_limit', 2.0),
            tileGridSize=self.params.get('clahe_tile_grid_size', (8, 8))
        )
        contrast_enhanced = clahe.apply(flattened)
        self.debug_data['clahe'] = contrast_enhanced

        d = self.params.get('bilateral_d', 5)
        sigma_color = self.params.get('bilateral_sigma_color', 50)
        sigma_space = self.params.get('bilateral_sigma_space', 5)
        smoothed = cv2.bilateralFilter(contrast_enhanced, d, sigma_color, sigma_space)
        self.debug_data['smoothed'] = smoothed

        string_map = smoothed

        # --- FRET PREPROCESS (lighter) ---
        fret_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(fret_gray, self.params['blur_kernel_size'], 0)

        # Strong vertical edge emphasis (Sobel Y)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.absolute(sobel_y)
        sobel_y = np.uint8(255* sobel_y / np.max(sobel_y))  # Normalize to 0-255
        self.debug_data['sobel_y'] = sobel_y

        median_intensity = np.median(fret_gray)
        std_intensity = np.std(fret_gray)
        self._log(f"Median intensity (fret path): {median_intensity}")

        if median_intensity < 100:
            canny_threshold1 = self.params['canny_threshold1_flat'] * std_intensity
            canny_threshold2 = self.params['canny_threshold2_flat'] * std_intensity
        else:
            canny_threshold1 = self.params['canny_threshold1_bright'] * std_intensity
            canny_threshold2 = self.params['canny_threshold2_bright'] * std_intensity

        contrast_enhanced = cv2.createCLAHE(
            clipLimit=self.params.get('clahe_clip_limit', 2.0),
            tileGridSize=self.params.get('clahe_tile_grid_size', (8, 8))
        ).apply(blur)

        self._log(f"Canny thresholds (fret path): {canny_threshold1}, {canny_threshold2}")
        edges = cv2.Canny(contrast_enhanced, int(canny_threshold1), int(canny_threshold2))
        self.debug_data['edges'] = edges

        return string_map, edges


    def _detect_features(self, roi_bgr: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
        """
        Detects string (horizontal) and fret (vertical) line positions in a fretboard ROI.
        Uses intensity profiling for strings and HoughLines for frets.
        """
        roi = roi_bgr.copy()
        h, w = roi.shape[:2]

        # --- STRING DETECTION: Horizontal Intensity Profiling ---
        string_map, edges = self._image_filter(roi)

        # Adaptive threshold for uneven lighting
        thresh = cv2.adaptiveThreshold(
            string_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Horizontal sum: bright strings → peaks
        horizontal_sum = np.sum(thresh, axis=1)

        # Smooth the signal
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_sum = np.convolve(horizontal_sum, kernel, mode='same')

        # Peak detection
        num_strings = 6  # set number of strings
        min_distance = h // (num_strings * 2)
        peaks, _ = find_peaks(smoothed_sum, distance=min_distance)

        # Adjust to exactly num_strings peaks
        peaks = sorted(peaks)
        if len(peaks) > num_strings:
            peak_values = [smoothed_sum[p] for p in peaks]
            top_indices = np.argsort(peak_values)[-num_strings:]
            peaks = sorted([peaks[i] for i in top_indices])
        elif len(peaks) < num_strings:
            string_height = h / num_strings
            existing_positions = set(peaks)
            for i in range(num_strings):
                est = int((i + 0.5) * string_height)
                if not any(abs(est - pos) < min_distance for pos in existing_positions):
                    peaks.append(est)
                    existing_positions.add(est)
            peaks = sorted(peaks)[:num_strings]

        string_positions = peaks
        self._log(f"Detected strings at: {string_positions}")

        # --- FRET DETECTION: HoughLines ---
        minLen_fret = int(max(10, h * self.params['fret_min_len_ratio']))
        lines_fret = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.params['fret_threshold'],
            minLineLength=minLen_fret,
            maxLineGap=self.params['fret_max_gap_ratio']
        )

        fret_positions = []
        if lines_fret is not None:
            for l in lines_fret:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                angle = angle if angle <= 90 else 180 - angle
                if angle > 45:  # vertical-ish
                    span = abs(y2 - y1)
                    if span >= int(h * self.params['fret_min_span_ratio']):
                        x_center = int((x1 + x2) / 2)
                        fret_positions.append(x_center)
                        self._log(f"Detected fret at x={x_center}")
                        self.debug_data.setdefault('raw_frets', []).append(l[0])

        # Cluster frets
        fret_positions = self._merge_lines(fret_positions, tol=self.params['fret_tolerance'])

        # --- Debug drawing ---
        debug = roi.copy()

        # Draw strings (green)
        for y in string_positions:
            cv2.line(debug, (0, y), (w - 1, y), (0, 255, 0), 1)

        # Draw frets (blue)
        for x in fret_positions:
            cv2.line(debug, (x, 0), (x, h - 1), (255, 0, 0), 1)

        return string_positions, fret_positions, debug


if __name__ == "__main__":
    # Example usage
    roi = cv2.imread("data/cropped_fretboard.jpg")
    detector = Strings_Frets(None, verbose=True)
    string_positions, fret_positions, debug_image = detector._detect_features(roi)
    cv2.imshow("Canny Edges", detector.debug_data.get('edges', np.zeros_like(roi)))
    cv2.imshow("Flattened", detector.debug_data.get('background_flattened', np.zeros_like(roi)))
    cv2.imwrite("data/canny_edges.jpg", detector.debug_data.get('edges', np.zeros_like(roi)))
    cv2.imshow("Debug", debug_image)
    cv2.imwrite("data/debug_image.jpg", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
