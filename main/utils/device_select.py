# Necessary Dependencies
import subprocess #Used for ffmpeg commands
import re #Used for discovering devices

class DeviceLister:
    def __init__(self):
        self.raw_output = ""
        self.video_devices = []
        self.audio_devices = []

    def list_devices(self):
        cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]

        # Capture stderr where FFmpeg prints device info
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.raw_output = result.stderr.decode('utf-8', errors='replace')

        print("Raw output")
        print(self.raw_output)

        self.video_devices = self._extract_devices("video")
        self.audio_devices = self._extract_devices("audio")

        return self.video_devices, self.audio_devices

    def _extract_devices(self, device_type):
        lines = self.raw_output.splitlines()
        devices = []

        for line in lines:
            if device_type in line.lower() and '"' in line:
                match = re.search(r'"(.*?)"', line)
                if match:
                    devices.append(match.group(1))
        return devices

    def get_default_devices():
        lister = DeviceLister()
        video_devices, audio_devices = lister.list_devices()

        video = video_devices[0] if video_devices else None
        audio = audio_devices[0] if audio_devices else None

        return video, audio

video, audio = DeviceLister.get_default_devices()
print("Video:", video)
print("Audio:", audio)