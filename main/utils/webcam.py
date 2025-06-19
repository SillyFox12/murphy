#Necessary dependencies
import csv
import os

import cv2 # Make sure opencv-python is installed: pip install opencv-python


#Extracts frames from the webcam and saves them as images 
class Webcam:
    def __init__(self, video_source, output_dir: str, csv_file: str):
        # Initialize the webcam with the given parameters
        self.video_source = video_source
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.ensure_directory(self.output_dir)
        print(f"Initializing Webcam with output_dir={output_dir}, csv_file={csv_file}")
    
    #Create output directory if it does not exist
    @staticmethod
    def ensure_directory(path=None):
        if not path:
            output_dir = "data"
            path = output_dir
        else:
            output_dir = path
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            raise

    #Capture frames from the webcam and save them as images. Alternatively, captures frames from a video file
    def capture_frames(self, video_source, output_dir: str, csv_file: str):
        #video_source can be a webcam index (0 for the default webcam) or a video file path
        
       # cap = cv2.VideoCapture(0 if video_source is None else video_source)

       # if not cap.isOpened():
       #     print(f"Error: Could not open video source {video_source}")
       #     return

    



       
