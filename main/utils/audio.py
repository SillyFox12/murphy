#Necessary dependencies
import os
import tempfile
import queue
import sys
from .dir_manager import DirectoryCreator as dc
# Necessary dependencies for audio extraction
from moviepy import VideoFileClip 
import sounddevice as sd
import soundfile as sf

# Extracts audio from a video file and saves it as a WAV file
class Microphone:

    # Initialize the Microphone class with video file and output directory
    def __init__(self, video_source, dir_manager: dc):
        self.video_source = video_source
        # Ensure the output directory exists
        self.output_dir = dir_manager.get_output_dir()
        print(f"Initializing Microphone with video_source={video_source}, output_dir={self.output_dir}")
        # Create a subdirectory for audio files
        self.audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

    # Set up the audio recording parameters
    def setup_audio(self):
        filename = f"recording_{next(tempfile._get_candidate_names())}.wav"
        self.filename = os.path.join(self.audio_dir, filename)

        self.channels = 1
        self.device = None  # default input
        self.subtype = 'PCM_16'
        self.samplerate = None  # Use the default samplerate of the input device
        self.q = queue.Queue()

    #Detects audio from the microphone and puts it into a queue
    def callback(self, indata, frames, time, status):
        """This function is called for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    #Records audio from the microphone and saves it to a file
    def extract_audio(self):
            print(f"Video source provided: {self.video_source}. Extracting audio from video.")
            #Load the video file
            video = VideoFileClip(self.video_source)
            # Extract audio from the video file
            audio = video.audio
            # Save the audio to a WAV file
            audio_file_path = os.path.join(self.audio_dir, 'extracted_audio.wav')
            audio.write_audiofile(audio_file_path, codec='pcm_s16le')
            print(f"Audio extracted and saved to {audio_file_path}")
    
    # Stops the audio recording
    def stop(self):
        print("\nRecording stopped. File saved as:", self.filename)

