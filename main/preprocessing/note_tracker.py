import os
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

class AnalysisConfig:
    def __init__(
        self,
        frame_analysis_interval: int = 5,
        audio_sr: int = 22050,
        audio_hop_length: int = 1024,
        audio_frame_size: int = 4096,
        audio_fmin: float = librosa.note_to_hz("C2"),
        audio_fmax: float = librosa.note_to_hz("C7"),
        fps: int = 30,
    ):
        if frame_analysis_interval < 1:
            raise ValueError("frame_analysis_interval must be 1 or greater.")
        self.frame_analysis_interval = frame_analysis_interval
        self.audio_sr = self.sr = audio_sr
        self.audio_hop_length = self.hop_length = audio_hop_length
        self.audio_frame_size = self.frame_size = audio_frame_size
        self.audio_fmin = self.fmin = audio_fmin
        self.audio_fmax = self.fmax = audio_fmax
        self.fps = fps

class Utilities:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def load(self, file: str) -> Tuple[bool, np.ndarray]:
        y, _ = librosa.load(file, sr=self.config.sr, mono=True, dtype=np.float32)
        return True, y

    def preprocess(self, y: np.ndarray, model_trim: bool = False) -> np.ndarray:
        y, _ = librosa.effects.trim(y, top_db=40)
        y -= np.mean(y)
        print(f"Trimmed audio length: {len(y)} samples")
        if model_trim:
            target = 16000
            if len(y) < target:
                y = np.pad(y, (0, target - len(y)))
            else:
                y = y[:target]
            print(f"Model-trimmed audio length: {len(y)} samples")
        rms = np.sqrt(np.mean(y**2) + 1e-12)
        return y / rms

    def stream_audio_dataset(self, data_path: str, batch_size: int = 4):
        files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(data_path)
            for f in fs
            if f.endswith(".wav")
        ]
        np.random.shuffle(files)
        for i in range(0, len(files), batch_size):
            batch = []
            for path in files[i : i + batch_size]:
                tracker = NoteTracker(self.config)
                _, y = tracker.load(path)
                batch.append(tracker.preprocess(y))
            yield np.array(batch)

    def mel_spectrogram(self, y: np.ndarray, sr: int, n_mels: int = 128, hop_length: int = 512):
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        return librosa.power_to_db(mel, ref=np.max)

    def visualize(self, y: np.ndarray):
        spec = self.mel_spectrogram(y, sr=self.config.sr, hop_length=self.config.hop_length)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spec, sr=self.config.sr, x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.show()

    def play(self, y: np.ndarray):
        print("Starting playback...")
        sd.play(y, self.config.sr)
        sd.wait()
        sd.stop()
        print("Playback finished.")
    
    def save_to_csv(self, events: list, output_path: str):
        """
        Saves the analysis events to a CSV file.
        """
        import csv
        
        if not events:
            print("No events to save.")
            return

        # Define headers (flattening the nested 'pitch' dict)
        headers = ["onset", "duration", "note", "freq", "cents"]

        try:
            with open(output_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                for ev in events:
                    # Flatten the data structure for CSV
                    row = {
                        "onset": f"{ev['onset']:.3f}",
                        "duration": f"{ev.get('duration', 0):.3f}", # Handle cases where duration might be missing
                        "note": ev["pitch"]["note"],
                        "freq": f"{ev["pitch"]["freq"]:.2f}",
                        "cents": f"{ev["pitch"]["cents"]:.2f}",
                    }
                    writer.writerow(row)
            print(f"Successfully saved analysis to: {output_path}")
        except IOError as e:
            print(f"Could not save CSV: {e}")

class NoteTracker(Utilities):
    def __init__(self, config: Optional[AnalysisConfig] = None):
        super().__init__(config)

    def freq_to_note(self, freq: float):
        if freq <= 0:
            return "N/A", 0.0
        midi = librosa.hz_to_midi(freq)
        midi_int = int(round(midi))
        note = librosa.midi_to_note(midi_int, octave=True)
        cents = 100 * (midi - midi_int)
        return note, cents

    def _multiband_flux(self, y, sr, n_fft=2048, hop_length=256, n_bands=4):
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_log = np.log1p(S)
        flux_bands = np.zeros(S_log.shape[1] - 1)
        edges = np.linspace(0, S_log.shape[0], n_bands + 1, dtype=int)
        for i in range(n_bands):
            band = S_log[edges[i]:edges[i+1], :]
            diff = np.diff(band, axis=1)
            diff = np.maximum(diff, 0.0)
            env = diff.mean(axis=0)
            weight = 1.0 + (i / n_bands)
            flux_bands += weight * env
        flux_bands = flux_bands / (flux_bands.max() + 1e-6)
        frame_times = librosa.frames_to_time(np.arange(len(flux_bands)), sr=sr, hop_length=hop_length)
        return flux_bands, frame_times

    def detect_onsets(self, y, sr=None) -> list:
        """
        Onset detection using harmonic-percussive source separation (HPSS)
        """
        sr = sr if sr is not None else self.config.sr

        # Padding to stabilize tail STFT frames (20–30 ms)
        pad = np.zeros(int(0.03 * sr))
        y = np.concatenate([y, pad])

        # 1. HPSS – isolate pluck energy
        _, y_perc = librosa.effects.hpss(y)

        # 2. Multi-band flux
        onset_env, frame_times = self._multiband_flux(y_perc, sr)

        # 3. Adaptive thresholding
        th = float(np.median(onset_env) + 0.05)

        # 4. Peak picking
        WIN = 3  # 3 frames ≈ 20–30 ms
        onset_frames = []

        for i in range(WIN, len(onset_env) - WIN):
            local = onset_env[i-WIN:i+WIN+1]
            if onset_env[i] == local.max() and onset_env[i] > th:
                onset_frames.append(i)
        onset_times = frame_times[onset_frames].tolist()

        # Ensure first onset is close to start
        if (len(onset_times)) > 0 and onset_times[0] > 0.05:
            first_idx = np.argmax(onset_env[:4])  # first few frames
            onset_times.insert(0, float(frame_times[first_idx]))

        # Ensure last onset near end
        TAIL_FRAMES = 12  # ~70–90 ms depending on hop_length
        tail = onset_env[-TAIL_FRAMES:]
        tail_times = frame_times[-TAIL_FRAMES:]

        peak_i = np.argmax(tail)
        last_time = float(tail_times[peak_i])
        last_val = float(tail[peak_i])

        # Conditions (research-backed)
        strong_enough = last_val > 0.25 * onset_env.max()  # lower threshold
        far_enough = (len(onset_times) == 0) or (last_time - onset_times[-1] > 0.05)
        rising_slope = (peak_i > 0 and tail[peak_i] > tail[peak_i - 1] * 1.1)

        if strong_enough and far_enough and rising_slope:
            onset_times.append(last_time)

        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=self.config.sr, alpha=0.6)
        for t in onset_times:
            plt.axvline(t, color="r", linestyle="--", linewidth=1.5)
        plt.title("Waveform with librosa-detected onsets")
        plt.tight_layout()
        plt.show()  
        
        return onset_times

    def adaptive_threshold(self, onset_env, w=16, offset=0.03) -> np.ndarray:
        """
        Local adaptive threshold:
        threshold[i] = median(onset_env[i-w:i+w]) + offset
        """
        from scipy.ndimage import median_filter

        med = median_filter(onset_env, size=w)
        return med + offset

    def monophonic_f0(self, y: np.ndarray):
        """
        Using pYIN. Consider tuning frame_length if latency is an issue.
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            sr=self.config.sr,
            frame_length=self.config.frame_size,
            hop_length=self.config.hop_length,
            center=True,
        )
        f0_times = librosa.frames_to_time(
            np.arange(len(f0)), sr=self.config.sr, hop_length=self.config.hop_length
        )
        return f0_times, f0, voiced_flag, voiced_probs

    def pitch_at_onsets(self, 
                        onset_times: list, 
                        f0_times: np.ndarray, 
                        f0_hz: np.ndarray, 
                        voiced_flag: np.ndarray):
        events = []
        onset_times = sorted(onset_times)

        # Define maximum analysis window (seconds)
        max_analysis_window = 0.2 

        for i, onset in enumerate(onset_times):
            # Define window: Start at onset, end at next onset OR max_analysis_window
            t_start = onset
            if i < len(onset_times) - 1:
                dist_to_next = onset_times[i+1] - onset
                duration = min(dist_to_next, max_analysis_window)
            else:
                duration = max_analysis_window     
            t_end = t_start + duration
            
            # Extract pitch candidates in this window
            mask = (f0_times >= t_start) & (f0_times <= t_end) & voiced_flag 
            if not np.any(mask):
                continue    
            f0_segment = f0_hz[mask]
            
            # Use median to ignore outliers
            f0_med = float(np.median(f0_segment))
            
            note, cents = self.freq_to_note(f0_med)

            events.append({
                "onset": float(onset),
                "duration": float(duration),
                "pitch": {
                    "freq": float(f0_med),
                    "note": note,
                    "cents": float(cents),
                },
            })   
        return events
    
    def group_pitches_by_time(self, detections: list):
        buckets = {}
        for d in detections:
            t = d["time"]
            buckets.setdefault(t, []).append(d)
        return buckets

    def analyze(self, y: np.ndarray):
        # 1. Detect Onsets
        onsets = self.detect_onsets(y, sr=self.config.sr)
        
        # 2. Detect Pitch (pYIN)
        f0_times, f0_hz, voiced_flag, _ = self.monophonic_f0(y)
        
        # 3. Align Pitch to Onsets
        events = self.pitch_at_onsets(onsets, f0_times, f0_hz, voiced_flag)
        
        # 4. (Optional) Frame indices calculation fixed
        frame_indices = [int(t * self.config.fps) for t in onsets]
        
        return events

if __name__ == "__main__":
    util = Utilities()
    tracker = NoteTracker(util.config)
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    filename = "data/audio_tests/quarter_cmaj.wav"
    loaded, music = util.load(filename)
    
    if not loaded:
        print(f"Failed to load audio file: {filename}")
    else:
        print("Audio file loaded successfully.")
        try:
            # 1. Preprocess
            processed = util.preprocess(music)
            
            # 2. Analyze (using the fixed analyze method from previous step)
            aligned_events = tracker.analyze(processed)
            
            # 3. Print results to console
            print(f"\nDetected {len(aligned_events)} events:")
            print("-" * 40)
            for ev in aligned_events:
                onset = ev["onset"]
                p = ev["pitch"]
                print(f"T={onset:.3f}s | {p['note']:<4} | {p['freq']:6.2f} Hz")
            
            # 4. Save to CSV
            csv_name = f"data/audio_analysis/{os.path.basename(filename).replace('.wav', '.csv')}"
            util.save_to_csv(aligned_events, csv_name)

            # 5. Visualize
            util.visualize(processed)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred during analysis: {e}")