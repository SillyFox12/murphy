# ------------------------- #
# main_pipeline_optimized.py
# ------------------------- #

import os
import sys
import cv2
import time
from multiprocessing import Process
from typing import List, Dict, Any

# Assume your custom modules are imported here.
# Mocks are used for demonstration.
from utils import AVRecorder, Microphone, DirectoryCreator, DeviceLister
from preprocessing import HandTracker, FeatureLogger, HandPoseFeatureEngineer, AudioAnalyzer, AnalysisConfig


class PerformanceAnalyzer:
    """
    An optimized, parallel pipeline for recording and analyzing performances.
    """
    def __init__(self, filename: str, duration: int, base_dir: str = "./data", config: AnalysisConfig = AnalysisConfig()):
        self.filename = filename
        self.duration = duration
        self.config = config
        self.dir_mgr = DirectoryCreator(base_dir=base_dir)
        self.video_path = os.path.join(self.dir_mgr.get_output_dir(), f"{self.filename}.mp4")

    def _record_performance(self) -> None:
        # Record the performance using the AVRecorder.
        print("--- [STEP 1] Recording Performance ---")
        video_device, audio_device = DeviceLister.get_default_devices()
        recorder = AVRecorder(
            video_device=video_device,
            audio_device=audio_device,
            dir_manager=self.dir_mgr,
            duration=self.duration
        )
        recorder.record(filename=self.filename)
        print(f"[✅] Performance recorded to: {self.video_path}")


    def _run_audio_analysis_process(self) -> None:
        # Extract audio and analyze it in a separate process.
        pid = os.getpid() # Get the process ID for logging
        print(f"[Audio Process PID: {pid}] Starting audio analysis...")

        mic = Microphone(video_source=self.video_path, dir_manager=self.dir_mgr)
        mic.extract_audio()
        audio_path = os.path.join(mic.get_output_dir(), "extracted_audio.wav")
        print(f"[Audio Process PID: {pid}] Audio extracted.")

        # Initialize the AudioAnalyzer with the provided configuration.
        analyzer = AudioAnalyzer(config=self.config)
        analysis_results = analyzer.analyze_audio(audio_path)

        # Export the analysis results to a CSV file.
        output_csv = os.path.join(mic.get_output_dir(), "pitch_chord_analysis.csv")
        analyzer.export_to_csv(analysis_results, output_path=output_csv)
        print(f"[Audio Process PID: {pid}] Analysis complete.")

    def _run_video_analysis_process(self) -> None:
        # Analyze the video in a separate process.
        # This process will only analyze frames at specified intervals to optimize performance.
        pid = os.getpid() # Get the process ID for logging
        print(f"[Video Process PID: {pid}] Starting optimized video analysis...")

        # Reads the vide file and processes it frame by frame in the system's memory.
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Video Process PID: {pid}] [ERROR] Could not open video: {self.video_path}", file=sys.stderr)
            return

        tracker = HandTracker()
        logger = FeatureLogger(output_dir=self.dir_mgr.get_output_dir(), filename="hand_features.csv")
        
        frame_num = 0
        processed_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process the frame if it's on the specified interval.
                if frame_num % self.config.frame_analysis_interval == 0:
                    frame, results = tracker.process_frame(frame)
                    logger.log_features(frame_number=frame_num, results=results)
                    processed_count += 1

                
                frame_num += 1
            
            total_frames = frame_num
            print(f"[Video Process PID: {pid}] Analyzed {processed_count} frames out of {total_frames} total.")
            print(f"[Video Process PID: {pid}] Analysis complete. Results saved to {logger.output_path}")

        finally:
            cap.release()
            tracker.cleanup()

    def run_pipeline(self) -> None:
        """Executes the entire pipeline."""
        start_time = time.time()
        
        self._record_performance()
        
        print("\n--- [STEP 2] Starting Parallel Optimized Analysis ---")

        audio_process = Process(target=self._run_audio_analysis_process)
        video_process = Process(target=self._run_video_analysis_process)

        audio_process.start()
        video_process.start()
        audio_process.join()
        video_process.join()
        
        end_time = time.time()
        print(f"\n[✅ DONE] Feedback system completed successfully in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    FILENAME = "test_performance"
    DURATION_SECONDS = 5
    
    # Create a config with our desired optimizations
    # Let's target a 5x reduction in video frames and use a lower audio sample rate.
    optimized_config = AnalysisConfig(
        frame_analysis_interval=1,
        audio_sr=16000
    )

    pipeline = PerformanceAnalyzer(
        filename=FILENAME,
        duration=DURATION_SECONDS,
        config=optimized_config
    )
    pipeline.run_pipeline()