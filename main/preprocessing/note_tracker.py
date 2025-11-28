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
    ):
        if frame_analysis_interval < 1:
            raise ValueError("frame_analysis_interval must be 1 or greater.")
        self.frame_analysis_interval = frame_analysis_interval
        self.audio_sr = self.sr = audio_sr
        self.audio_hop_length = self.hop_length = audio_hop_length
        self.audio_frame_size = self.frame_size = audio_frame_size
        self.audio_fmin = self.fmin = audio_fmin
        self.audio_fmax = self.fmax = audio_fmax


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


class NoteTracker(Utilities):
    def __init__(self, config: Optional[AnalysisConfig] = None):
        super().__init__(config)

    def freq_to_note(self, freq: float):
        midi = librosa.hz_to_midi(freq)
        midi_int = int(round(midi))
        note = librosa.midi_to_note(midi_int, octave=True)
        cents = 100 * (midi - midi_int)
        return note, cents

    def _parabolic_interpolation(self, mag: np.ndarray, idx: int, freqs: np.ndarray):
        if idx <= 0 or idx >= len(mag) - 1:
            return float(freqs[idx]), float(mag[idx])
        a, b, g = map(float, (mag[idx - 1], mag[idx], mag[idx + 1]))
        denom = a - 2 * b + g
        if denom == 0 or not np.isfinite(denom):
            return float(freqs[idx]), b
        p = 0.5 * (a - g) / denom
        if not np.isfinite(p):
            p = 0.0
        p = max(min(p, 1.0), -1.0)
        bin_hz = float(freqs[1] - freqs[0])
        f = float(freqs[idx] + p * bin_hz)
        m = float(b - 0.25 * (a - g) * p)
        if not np.isfinite(f):
            f = float(freqs[idx])
        if not np.isfinite(m):
            m = b
        return f, m

    def _hps_for_vector(self, mag: np.ndarray, n_harmonics: int = 5, eps: float = 1e-12):
        mag = np.maximum(mag, eps)
        log_mag = np.log(mag)
        hps = log_mag.copy()
        for k in range(2, n_harmonics + 1):
            dec = log_mag[::k]
            if len(dec) < len(hps):
                pad_len = len(hps) - len(dec)
                dec = np.pad(dec, (0, pad_len), constant_values=np.log(eps))
            hps[: len(dec)] += dec[: len(hps)]
        return np.exp(hps - hps.max())

    def detect_onsets(self, y: np.ndarray):
        frames = librosa.onset.onset_detect(
            y=y,
            sr=self.config.sr,
            hop_length=self.config.hop_length,
            backtrack=True,
        )
        times = librosa.frames_to_time(frames, sr=self.config.sr, hop_length=self.config.hop_length)
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=self.config.sr, alpha=0.6)
        for t in times:
            plt.axvline(t, color="r", linestyle="--", linewidth=1.5)
        plt.title("Waveform with librosa-detected onsets")
        plt.tight_layout()
        plt.show()
        return times

    def polyphonic_hps(
        self,
        y: np.ndarray,
        sr: Optional[int] = None,
        n_harmonics: int = 5,
        threshold_rel: float = 0.08,
        rms_threshold: float = 1e-4,
        max_peaks: int = 6,
    ) -> list:
        sr = sr or self.config.sr
        n_fft = self.config.frame_size
        hop = self.config.hop_length
        fmin, fmax = self.config.fmin, self.config.fmax
        if y is None or len(y) == 0:
            return []
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mask = (freqs >= fmin) & (freqs <= fmax)
        S_crop = S[mask]
        freqs_crop = freqs[mask]
        times = librosa.frames_to_time(np.arange(S_crop.shape[1]), sr=sr, hop_length=hop)
        det = []
        for ti in range(S_crop.shape[1]):
            mag = S_crop[:, ti]
            if np.sqrt(np.mean(mag**2)) < rms_threshold:
                continue
            hps = self._hps_for_vector(mag, n_harmonics=n_harmonics)
            thr = threshold_rel * np.max(hps)
            if thr <= 0:
                continue
            try:
                peaks = librosa.util.peak_pick(hps, 3, 3, 3, 3, thr, 5)
            except Exception:
                peaks = np.where(hps > thr)[0]
            if len(peaks) == 0:
                continue
            sal = hps[peaks]
            peaks = peaks[np.argsort(-sal)][:max_peaks]
            for idx in peaks:
                f_interp, s_interp = self._parabolic_interpolation(hps, idx, freqs_crop)
                if not np.isfinite(f_interp) or f_interp <= 0:
                    continue
                note, cents = self.freq_to_note(f_interp)
                det.append(
                    {
                        "time": float(times[ti]),
                        "freq": float(f_interp),
                        "salience": float(s_interp),
                        "note": note,
                        "cents": float(cents),
                    }
                )
        return det

    def group_pitches_by_time(self, detections: list):
        buckets = {}
        for d in detections:
            t = d["time"]
            buckets.setdefault(t, []).append(d)
        return buckets

    def pitch_at_onsets(self, onset_times: list, pitch_detections: list, window: float = 0.05):
        events = []
        for onset in onset_times:
            cand = [d for d in pitch_detections if onset <= d["time"] <= onset + window]
            if not cand:
                continue
            cand.sort(key=lambda d: -d["salience"])
            events.append({"onset": onset, "pitch": cand[0]})
        return events

    def analyze(self, y: np.ndarray):
        pitches = self.polyphonic_hps(y)
        onsets = self.detect_onsets(y)
        return self.pitch_at_onsets(onsets, pitches)


if __name__ == "__main__":
    util = Utilities()
    tracker = NoteTracker(util.config)
    loaded, music = util.load("data/test.wav")
    if not loaded:
        print("Failed to load audio file.")
    else:
        print("Audio file loaded successfully.")
        try:
            processed = util.preprocess(music)
            aligned = tracker.analyze(processed)
            for ev in aligned:
                onset = ev["onset"]
                p = ev["pitch"]
                print(f"Onset {onset:.3f}s → {p['note']} ({p['freq']:.2f} Hz)")
            util.visualize(processed)
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
