#Necessary dependencies
import os

import cv2 # Make sure opencv-python is installed: pip install opencv-python


#Extracts frames from the webcam and saves them as images 
class Webcam:
    def __init__(self, video_source, output_dir: str):
        # Initialize the webcam with the given parameters
        self.video_source = video_source
        self.output_dir = output_dir
        print(f"Initializing Webcam with video_source={video_source}, output_dir={output_dir}")
        self.ensure_directory(self.output_dir)

    #Create output directory if it does not exist
    @staticmethod
    def ensure_directory(path=None):
        if not path:
            output_dir = "./data"
            path = output_dir
        else:
            output_dir = path
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            raise

    #Decides whether to use the webcam or a video file as the source
    def get_video_capture(self):
        #video_source can be a webcam index (0 for the default webcam) or a video file path
        cap = 0 if self.video_source is None else self.video_source
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
        output_dir = self.output_dir
        current_frame = 0
        print(f"Extracting frames from {self.video_source} and saving them to {self.output_dir}")

        #Pulls out the frames from the video source
        while True:
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

                cv2.putText(frame, "Press 'e' to exit the live feed.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Frame", frame)

                # Wait for 1 ms and check if 'e' is pressed to exit
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    break
            else:
                print("No more frames to read or an error occurred.")
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        cap = self.get_video_capture()
        self.frame_extractor(cap)

cam = Webcam(video_source="C:\\Users\\My_ka\\code\\murphy\\main\\training\\test.mp4", output_dir="data")  # Use webcam
cam.run()

# OR for a video file:
# cam = Webcam(video_source="my_video.mp4", output_dir="data")
# cam.run()


