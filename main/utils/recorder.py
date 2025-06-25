import subprocess
import os
from dir_manager import DirectoryCreator
from device_select import DeviceLister

class AVRecorder:
    #Initialize arguments
    def __init__(self, video_device: DeviceLister, audio_device: DeviceLister, dir_manager : DirectoryCreator, framerate=30, duration=None):
        self.video_device, self.audio_device = DeviceLister.get_default_devices()
        self.output_dir = dir_manager.get_output_dir()
        self.framerate = framerate
        self.duration = duration  # in seconds (optional)
    #Variables for video
    def build_command(self, filename: str):
        video_input = f'video="{self.video_device}"'
        audio_input = f'audio="{self.audio_device}"'
        output_file = os.path.join(self.output_dir, f"{filename}.mp4")
        #Arra containing the elements of the ffmpeg command.
        cmd = [
            "ffmpeg",
            "-f", "dshow",
            "-i", f"{video_input}:{audio_input}",
            "-t", str(self.duration),
            "-c:v", "libx264",
            "-c:a", "pcm_s16le", #WAV audio
            "-strict", "experimental",
            "-pix_fmt", "yuv420p",
            output_file
        ]

        return cmd
    #Saves the recording as a file
    def record(self, filename="output"):
        command = self.build_command(filename)
        print("Running FFmpeg command:")
        print(" ".join(command))
        subprocess.run(" ".join(command), shell=True)

if __name__ == "__main__":
   dir_mgr = DirectoryCreator(output_dir="./data")
   video, audio = DeviceLister.get_default_devices()
   recorder = AVRecorder(
       video_device=video,
       audio_device=audio,
       dir_manager=dir_mgr,
       framerate=30,
       duration=10
   )
   recorder.record(filename="test_recording")