# ------------------------- #
# main_pipeline.py
# ------------------------- #

# Necessary dependencies
import sys
import os
import glob
import cv2

# Custom utility modules
import utils
from utils import AVRecorder, Microphone, DirectoryCreator, DeviceLister, FrameExtractor


# Preprocessing modules
import preprocessing
from preprocessing import HandTracker, LandmarkLogger

# Record a performance using AVRecorder
def record_performance(filename="performance", duration=10):
    global dir_mgr
    dir_mgr = DirectoryCreator(base_dir="./data")
    video_device, audio_device = DeviceLister.get_default_devices()

    recorder = AVRecorder(
        video_device=video_device,
        audio_device=audio_device,
        dir_manager=dir_mgr,
        duration=duration
    )
    recorder.record(filename=filename)
    return os.path.join(dir_mgr.get_output_dir(), f"{filename}.mp4"), dir_mgr

# Extract audio from the video using Microphone class
def extract_audio(video_path, dir_mgr):
    mic = Microphone(video_source=video_path, dir_manager=dir_mgr)
    mic.extract_audio()
    return os.path.join(mic.get_output_dir(), "extracted_audio.wav")

def extract_frames(video_path, dir_mgr):
    video = FrameExtractor(video_source=video_path, dir_manager=dir_mgr)
    video.record()

#STEP 3: Analyze the saved frames using MediaPipe
def analyze_video():
    video = FrameExtractor(video_source=None, dir_manager=dir_mgr)
    image_paths = sorted(glob.glob(f'{video.get_output_dir()}/*.jpg'))
    tracker = HandTracker()
    logger = LandmarkLogger(output_dir=video.get_output_dir())
    logger.write_header()

    print(f"[INFO] Analyzing {len(image_paths)} frames for hand posture...")

    for frame_num, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue

        frame, results = tracker.process_frame(frame)
        tracker.display_frame(frame, results)

        logger.save_landmarks(frame_num, results)

        if tracker.exit():
            break

    tracker.cleanup()
    print(f"[INFO] Finished analyzing {len(image_paths)} frames. Saved to {logger.filepath}")

#Main execution pipeline
def main():
    print("[STEP 1] Recording performance...")
    video_path, dir_mgr = record_performance(filename="test_recording", duration=10)

    print("[STEP 2] Extracting audio from recording...")
    audio_path = extract_audio(video_path, dir_mgr)

    print("[Step 3] Extracting the frames from the recording...")
    extract_frames(video_path, dir_mgr)
    
    print("[STEP 3] Analyzing hand movement and posture...")
    analyze_video()

    print("[✅ DONE] Feedback system completed successfully.")

# Python entry point
if __name__ == "__main__":
    main()

