from concurrent.futures import wait
import os
import librosa
import sounddevice as sd
from scipy import signal
import noisereduce as nr

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

class AnalysisConfig:
    """A centralized configuration object for the entire analysis pipeline."""
    def __init__(self,
                 frame_analysis_interval: int = 5,
                 # --- Audio Parameters ---
                 audio_sr: int = 22050,
                 audio_hop_length: int = 1024,
                 audio_frame_size: int = 4096,
                 audio_fmin: float = librosa.note_to_hz('C2'),
                 audio_fmax: float = librosa.note_to_hz('C7')):
        """
        Args:
            frame_analysis_interval (int): Interval for video frame analysis.
            audio_sr (int): Sample rate for audio analysis. Lower is faster.
            audio_hop_length (int): Hop length for STFT. Higher is faster.
            audio_frame_size (int): FFT window size.
            audio_fmin (float): Minimum frequency for pitch detection (pYIN).
            audio_fmax (float): Maximum frequency for pitch detection (pYIN).
        """
        if frame_analysis_interval < 1:
            raise ValueError("frame_analysis_interval must be 1 or greater.")
        
        # --- Core Attributes ---
        self.frame_analysis_interval = frame_analysis_interval # Interval for video frame analysis
        self.audio_sr = audio_sr # Sample rate for audio analysis
        self.audio_hop_length = audio_hop_length # Hop length for STFT
        self.audio_frame_size = audio_frame_size # FFT window size
        self.audio_fmin = audio_fmin # Hz for pitch detection lower limit
        self.audio_fmax = audio_fmax # Hz for pitch detection upper limit
        
        # --- Aliases for librosa compatibility ---
        # This simplifies calls to librosa functions.
        self.sr = self.audio_sr
        self.hop_length = self.audio_hop_length
        self.frame_size = self.audio_frame_size
        self.fmin = self.audio_fmin
        self.fmax = self.audio_fmax

class Utilities:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config if config is not None else AnalysisConfig()
            
    
    def load(self, file) -> Tuple[bool, np.ndarray]:
       """Load an audio file"""
       loaded = False
       y, _ = librosa.load(file, sr=self.config.sr, mono=True, dtype=np.float32)
       loaded = True

       return loaded, y

    def preprocess(self, y, model_trim=False) -> np.ndarray:
        """Preprocess the audio data""" 
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        print(f"Trimmed audio length: {len(y_trimmed)} samples")

        if model_trim:
            # Trim the length of the signal to model requirements
            target_length = 16000 # Example target length

            if len(y_trimmed) < target_length:
                y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
            else:
                y_trimmed = y_trimmed[:target_length]
            
            print(f"Model-trimmed audio length: {len(y_trimmed)} samples")
            return y_trimmed

        # High-pass filter to remove low-frequency noise    
        highpass = True
        cutoff_freq = 50 # Cutoff frequency in Hz

        allpass_out = np.zeros_like(y_trimmed) # Output array for the all-pass filter
        dn_1 = 0.0 # Delay element for the filter

        for n in range(len(y_trimmed)):
            break_freq = cutoff_freq / (self.config.sr / 2) # Normalize cutoff frequency

            tan = np.tan(np.pi * break_freq)

            a1 = (tan - 1) / (tan + 1)
            allpass_out[n] = a1 * y_trimmed[n] + dn_1
        
        if highpass:
            allpass_out *= -1
        
        highpassed = y_trimmed + allpass_out
        
        # Reduce noise in audio with spectral gating
        y_denoised = nr.reduce_noise(y=highpassed, sr=self.config.sr)
        
        # Normalize the audio signal
        y_denoised = librosa.util.normalize(y_denoised)

        return y_denoised
    
    def stream_audio_dataset(self, data_path, batch_size=4):
        """Stream audio dataset in batches"""
        # Obtain list of audio files
        audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(data_path) for file in files if file.endswith('.wav')]
        # Randomly shuffle the audio files for each epoch
        np.random.shuffle(audio_files)
        # Yield batches of audio files
        for i in range(0, len(audio_files), batch_size):
            batch_paths = audio_files[i:i + batch_size]
            batch_data = []

            for file_path in batch_paths:
                tracker = NoteTracker(self.config)
                is_loaded, y = tracker.load(file_path)
                y_processed = tracker.preprocess(y)
                batch_data.append(y_processed)

            yield np.array(batch_data)
    
    def mel_spectrogram(self, y, sr, n_mels=128, hop_length=512):
        """Compute the Mel spectrogram of the audio"""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
        
    
    def visualize(self, y):
        """Visualize the audio"""
        # Generate the spectrogram
        processor = Utilities()
        spec = processor.mel_spectrogram(y, sr=self.config.sr, hop_length=self.config.hop_length)

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spec, sr=self.config.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.show()

    def play(self, y):
        """Play an audio file from time series data"""
        # Play the audio using sounddevice
        print("Starting playback...")
        sd.play(y, self.config.sr)
        sd.wait() 
        sd.stop()
        print("Playback finished.") 

class NoteTracker(Utilities):

    # 60 exact frequencies from C2 to B6 (A440 equal temperament)
    HPS_NOTE_FREQS = librosa.cqt_frequencies(
        n_bins=60,
        fmin=librosa.note_to_hz('C2'),   # exact C2 frequency
        bins_per_octave=12
    )  # dtype=float64, perfect precision

    # Corresponding note names (you can choose sharps or flats)
    NOTE_NAMES = [librosa.midi_to_note(midi, octave=True) 
                for midi in range(36, 96)]

    # Optional: flats version
    NOTE_NAMES_FLAT = [librosa.midi_to_note(midi, octave=True, key='b') 
                    for midi in range(36, 96)]

    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        super().__init__(config)
        self.config = config if config is not None else AnalysisConfig()
    
     # Grok
    def freq_to_note(self, freq: float, flats: bool = False) -> str:
        idx = np.argmin(np.abs(self.HPS_NOTE_FREQS - freq))
        return self.NOTE_NAMES_FLAT[idx] if flats else self.NOTE_NAMES[idx]
    
    # Also get cents deviation (very useful for tuning accuracy!)
    def freq_to_note_cents(self, freq: float) -> tuple[str, float]:
        idx = np.argmin(np.abs(self.HPS_NOTE_FREQS - freq))
        nearest_freq = self.HPS_NOTE_FREQS[idx]
        cents_off = 1200 * np.log2(freq / nearest_freq)
        return self.NOTE_NAMES[idx], cents_off
    
    def note_detector(self, y: np.ndarray, rms_threshold: float = 0.01, n_harmonics: int = 4, threshold_rel: float = 0.1) -> list:
        """Detect notes in the audio data"""
        # Skip silent frames
        if np.sqrt(np.mean(y**2)) < rms_threshold:
            return []
        
        # Pitch detection using HPS
        S = np.abs(librosa.stft(y, n_fft=self.config.frame_size, 
                                hop_length=self.config.hop_length))
        
        # "Build frequency axis"
        freqs = librosa.fft_frequencies(sr=self.config.sr, n_fft=self.config.frame_size)

        # 4. Crop to meaningful pitch range
        mask = (freqs >= self.config.fmin) & (freqs <= self.config.fmax)
        S_crop = S[mask]
        freqs_crop = freqs[mask]

        if S_crop.shape[0] == 0:
            return []

        # 5. Harmonic Product Spectrum (vectorized, no loops!)
        hps = S_crop.copy()
        for k in range(2, n_harmonics + 1):
            decimated = S_crop[::k]  # downsample by integer k
            pad_length = hps.shape[0] - decimated.shape[0]
            if pad_length > 0:
                decimated = np.pad(decimated, (0, pad_length), mode='constant')
            hps *= decimated[:hps.shape[0]]

        # 6. Dynamic threshold: relative to max peak
        threshold = threshold_rel * hps.max()

        # 7. Find all peaks above threshold
        peak_indices = np.where(hps > threshold)[0]
        if len(peak_indices) == 0:
            return []

        # 8. Convert to real frequencies and salience
        peak_freqs = freqs_crop[peak_indices]
        peak_salience = hps[peak_indices]

        # 9. Sort by salience (strongest first) — optional but standard
        sort_idx = np.argsort(-peak_salience)
        results = [(float(peak_freqs[i]), float(peak_salience[i])) 
                for i in sort_idx]

        return results

if __name__ == "__main__":
    util = Utilities()
    is_loaded, music = util.load("data/test.wav")

    if is_loaded:
        print("Audio file loaded successfully.")
    else:
        print("Failed to load audio file.")

    print(f"This is the numpy array of the audio file: {music}")
    processed_music = util.preprocess(music, True)

    try:
        util.play(processed_music)
        util.visualize(music)
        util.visualize(processed_music)
    except Exception as e:
        print(f"An error occurred during playback: {e}")

# Load the audiofile
# Resample
# Convert to mono
# Trim silence and noise
# Detect note onsets
# Detect pitch