#Necessary dependencies
import os
from .dir_manager import DirectoryCreator as dc
import cv2 # Make sure opencv-python is installed: pip install opencv-python

#Extracts frames from the webcam and saves them as images 
class FrameExtractor:
    def __init__(self, video_source, dir_manager: dc):
        # Initialize the webcam with the given parameters
        self.video_source = video_source
        # Use DirectoryCreator to ensure the output directory exists
        self.output_dir = dir_manager.get_output_dir()
        # Print the initialization parameters for debugging
        print(f"Initializing Webcam with video_source={video_source}, output_dir={self.output_dir}")
    
    #Decides whether to use the webcam or a video file as the source
    def get_video_capture(self):
        #video_source can be a webcam index (0 for the default webcam) or a video file path
        cap = self.video_source
        # If video_source is None, use the default webcam (0)
        cap = cv2.VideoCapture(cap)
        # Check if the default webcam is opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}. Please check the source and try again.")
            exit()
        video_type = "default webcam" if self.video_source is None else f"video file {self.video_source}"
        print(f"Using {video_type} as video source.")
        return cap

    #Extract frames from the video source and save them as images
    def frame_extractor(self, cap):
        # Capture frames from the webcam or video file
        self.running = True
        # Ensure the output directory exists
        output_dir = self.output_dir
        current_frame = 0
        print(f"Extracting frames from {self.video_source} and saving them to {self.output_dir}")

        #Pulls out the frames from the video source
        try:
            while self.running:
                # Reading from frame
                ret, frame = cap.read()

                if ret:
                    # Creates images if there is still video left.
                    name = f'./{output_dir}/frame{current_frame}.jpg'
                    print(f'Creating... {name}')

                    # Writing the extracted images
                    cv2.imwrite(name, frame)

                    # Increasing counter so that it will show how many frames are created
                    current_frame += 1

                    cv2.putText(frame, "Press 'Ctrl+C' to exit the live feed.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Frame", frame)

                    # Wait for 1 ms to display the frame
                    # Allows the window to be reactive to keyboard events
                    cv2.waitKey(1)
                else:
                    print("No more frames to read or an error occurred.")
                    break
        except:
            # Release the capture and close any OpenCV windows
            self.stop()
    # Stops the video capture and closes all OpenCV windows
    def stop(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            print("Video capture stopped.")
        cv2.destroyAllWindows()
        print("All OpenCV windows closed.")
        
    #Runs the webcam recording process
    def record(self):
        self.cap = self.get_video_capture()
        self.frame_extractor(self.cap)