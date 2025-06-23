#Necessary dependencies
import os
import tempfile
import queue
import sys
from dir_manager import DirectoryCreator as dc
# Necessary dependencies for audio extraction
#from moviepy.editor import VideoFileClip
import sounddevice as sd
import soundfile as sf

#Extracts audio from a video file and saves it as a WAV file
class Microphone:
    
    #Initialize the Microphone class with video file and output directory
    def __init__(self, video_source,  dir_manager: dc):
        self.video_source = video_source
        # Ensure the output directory exists
        self.output_dir = dir_manager.get_output_dir()
        print(f"Initializing Microphone with video_source={video_source}, output_dir={self.output_dir}")

    # Set up the audio recording parameters
    def setup_audio(self):
        filename = f"recording_{next(tempfile._get_candidate_names())}.wav"
        self.filename = os.path.join(self.output_dir, filename)

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
    def record(self):
        self.setup_audio()
        try:
            if self.samplerate is None:
                device_info = sd.query_devices(self.device, 'input')
                self.samplerate = int(device_info['default_samplerate'])

            with sf.SoundFile(self.filename, mode='x', samplerate=self.samplerate,
                              channels=self.channels, subtype=self.subtype) as file:
                with sd.InputStream(samplerate=self.samplerate, device=self.device,
                                    channels=self.channels, callback=self.callback):
                    print("Recording... Press Ctrl+C to stop.")
                    while True:
                        file.write(self.q.get())
        except KeyboardInterrupt:
            print("\nRecording stopped. File saved as:", self.filename)
        except Exception as e:
            print("An error occurred:", str(e))
        

mic = Microphone(video_source=None, dir_manager=dc())
mic.record()