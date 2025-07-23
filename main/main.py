# ------------------------- #
# main_pipeline_final.py
# ------------------------- #

import os
import csv
import sys
import cv2
import time
from multiprocessing import Process
from typing import List, Dict, Any, Optional

# --- MODULE IMPORTS ---
# Import all necessary, well-defined classes from your modules.
from utils import AVRecorder, Microphone, DirectoryCreator, DeviceLister
from preprocessing import (
    AnalysisConfig, AudioAnalyzer, HandTracker,
    HandPoseFeatureEngineer, GuitarNeckDetector
)

# ==============================================================================
# THE ORCHESTRATOR: A NEW CLASS FOR UNIFIED VISUAL FEATURE ENGINEERING
# ==============================================================================

class VisualFeaturePipeline:
    """
    A professional-grade pipeline that orchestrates all visual analysis,
    fusing hand posture and fretboard interaction data into a single,
    unified feature vector for machine learning.
    """
    def __init__(self, output_dir: str, filename: str = "visual_features.csv", verbose: bool = False):
        """
        Initializes all necessary visual processing components.
        """
        self.output_path = os.path.join(output_dir, filename)
        self.verbose = verbose

        # Instantiate all required engineering and detection modules.
        self.hand_pose_engineer = HandPoseFeatureEngineer()
        self.neck_detector = GuitarNeckDetector(verbose=self.verbose)
        
        # Define the complete, unified header for the output CSV.
        self.header = self._create_comprehensive_header()
        self._write_header_if_needed()

    def _log(self, message: str):
        """Logs a message if verbosity is enabled."""
        if self.verbose:
            print(f"[VisualPipeline] {message}")

    def _create_comprehensive_header(self) -> List[str]:
        """Dynamically creates the header for the output CSV file."""
        header = ["frame", "hand_index"]
        # Add all hand posture features.
        header.extend(self.hand_pose_engineer.get_feature_names())
        # Add all fretboard interaction features for each of the four fingers.
        for finger in ["index", "middle", "ring", "pinky"]:
            header.extend([
                f"{finger}_fret_idx",
                f"{finger}_string_idx",
                f"{finger}_fret_dist_px"
            ])
        return header

    def _write_header_if_needed(self):
        """Writes the CSV header if the file does not already exist."""
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writeheader()

    def process_and_log_frame(self, frame: np.ndarray, frame_number: int, hand_results):
        """
        The core method that processes a single frame and logs the unified feature vector.
        """
        # 1. First, attempt to track the fretboard in the current frame.
        fretboard_detected = self.neck_detector.track_fretboard(frame, frame_number)

        # 2. If no hands are detected, there is nothing to log.
        if not hand_results.multi_hand_landmarks:
            return

        # 3. For each detected hand, compute and log the full feature set.
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)

            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Start with a base row containing frame and hand index.
                row_data = {"frame": frame_number, "hand_index": hand_idx}
                
                # A. Compute the hand's internal posture features.
                pose_features = self.hand_pose_engineer.calculate_features(hand_landmarks)
                if pose_features:
                    row_data.update(pose_features)

                # B. If a fretboard is visible, compute interaction features.
                if fretboard_detected:
                    frame_h, frame_w, _ = frame.shape
                    # Get landmarks for the four primary fingers.
                    fingertip_indices = {
                        "index": 8, "middle": 12,
                        "ring": 16, "pinky": 20
                    }
                    for finger_name, lm_idx in fingertip_indices.items():
                        # De-normalize landmark coordinates to pixel space.
                        landmark = hand_landmarks.landmark[lm_idx]
                        pixel_coord = (int(landmark.x * frame_w), int(landmark.y * frame_h))
                        
                        # Get detailed grid information for the fingertip.
                        grid_details = self.neck_detector.get_grid_cell_details(pixel_coord)
                        if grid_details:
                            row_data[f"{finger_name}_fret_idx"] = grid_details["fret_idx"]
                            row_data[f"{finger_name}_string_idx"] = grid_details["string_idx"]
                            row_data[f"{finger_name}_fret_dist_px"] = grid_details["fret_dist_px"]
                
                # C. Write the complete, unified feature row to the CSV.
                writer.writerow(row_data)

# ==============================================================================
# THE MAIN APPLICATION: CLEAN, DECLARATIVE, AND PROFESSIONAL
# ==============================================================================

class PerformanceAnalyzer:
    """
    The main orchestrator for the performance analysis pipeline.
    This class is now cleaner, delegating complex visual logic to the VisualFeaturePipeline.
    """
    def __init__(self, filename: str, duration: int, base_dir: str = "./data", config: AnalysisConfig = AnalysisConfig(), verbose: bool = False):
        self.filename = filename
        self.duration = duration
        self.config = config
        self.verbose = verbose
        self.dir_mgr = DirectoryCreator(base_dir=base_dir)
        self.video_path = os.path.join(self.dir_mgr.get_output_dir(), f"{self.filename}.mp4")

    def _record_performance(self) -> None:
        print("--- [STEP 1] Recording Performance ---")
        video_device, audio_device = DeviceLister.get_default_devices()
        recorder = AVRecorder(video_device=video_device, audio_device=audio_device, dir_manager=self.dir_mgr, duration=self.duration)
        recorder.record(filename=self.filename)
        print(f"[✅] Performance recorded to: {self.video_path}")

    def _run_audio_analysis_process(self) -> None:
        pid = os.getpid()
        print(f"[Audio Process PID: {pid}] Starting audio analysis...")
        mic = Microphone(video_source=self.video_path, dir_manager=self.dir_mgr)
        mic.extract_audio()
        audio_path = os.path.join(mic.get_output_dir(), "extracted_audio.wav")
        analyzer = AudioAnalyzer(config=self.config)
        analysis_results = analyzer.analyze_audio(audio_path)
        output_csv = os.path.join(self.dir_mgr.get_output_dir(), "audio_features.csv")
        analyzer.export_to_csv(analysis_results, output_path=output_csv)
        print(f"[Audio Process PID: {pid}] Audio analysis complete.")

    def _run_video_analysis_process(self) -> None:
        """
        This process is now beautifully simple. It delegates all complex logic
        to the dedicated VisualFeaturePipeline.
        """
        pid = os.getpid()
        print(f"[Video Process PID: {pid}] Starting unified visual analysis...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Video Process PID: {pid}] [ERROR] Could not open video: {self.video_path}", file=sys.stderr)
            return

        # 1. Instantiate the trackers and the new, unified pipeline.
        hand_tracker = HandTracker()
        visual_pipeline = VisualFeaturePipeline(output_dir=self.dir_mgr.get_output_dir(), verbose=self.verbose)
        
        frame_num = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 2. Get hand landmarks from the frame.
                hand_results = hand_tracker.process_frame(frame)
                
                # 3. Pass the raw data to the pipeline for processing and logging.
                # All complex logic is now encapsulated in this single method call.
                visual_pipeline.process_and_log_frame(frame, frame_num, hand_results)
                
                frame_num += 1
            
            print(f"[Video Process PID: {pid}] Visual analysis complete. Results saved to {visual_pipeline.output_path}")

        finally:
            cap.release()
            hand_tracker.cleanup()

    def run_pipeline(self) -> None:
        """Executes the entire multi-process pipeline."""
        start_time = time.time()
        self._record_performance()
        print("\n--- [STEP 2] Starting Parallel Analysis (Audio & Visuals) ---")
        audio_process = Process(target=self._run_audio_analysis_process)
        video_process = Process(target=self._run_video_analysis_process)
        audio_process.start(); video_process.start()
        audio_process.join(); video_process.join()
        print(f"\n[✅ DONE] Full pipeline completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    FILENAME = "final_performance"
    DURATION_SECONDS = 5
    
    # The config now only needs to hold parameters for the AudioAnalyzer.
    # Visual parameters are managed within their respective classes.
    config = AnalysisConfig(audio_sr=16000)

    pipeline = PerformanceAnalyzer(
        filename=FILENAME,
        duration=DURATION_SECONDS,
        config=config,
        verbose=True  # Enable verbose logging for debugging.
    )
    pipeline.run_pipeline()