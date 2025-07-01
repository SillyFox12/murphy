import numpy as np
import librosa

def detect_pitch_hps(audio_path, sr=22050, harmonics=4, hop_length=256): # Looks for 4 harmonic frequencies including the fundamental.
    y, sr = librosa.load(audio_path, sr=sr)
    f0_series = []

    frame_len = 1024
    #Hanning frames helps to reduce aliasing when creating discrete sections
    window = np.hanning(frame_len)

    for i in range(0, len(y) - frame_len, hop_length):
        frame = y[i:i+frame_len] * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        hps = magnitude.copy()

        for h in range(2, harmonics + 1):
            dec = magnitude[::h]
            hps[:len(dec)] *= dec

        freqs = np.fft.rfftfreq(len(frame), d=1/sr)
        pitch_idx = np.argmax(hps)
        f0 = freqs[pitch_idx]
        f0_series.append(f0 if 50 < f0 < 1000 else 0)  # Sanity check

    return f0_series
