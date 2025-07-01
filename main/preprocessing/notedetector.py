import numpy as np
import librosa

class PitchReader:
    #Initializes necessary arguments
    def __init__(self, audio_path):
        self.audio_path = audio_path
        
    def detect_pitch_hps(self, sr=22050, harmonics=4, hop_length=256): # Looks for 4 harmonic frequencies including the fundamental.
        y, sr = librosa.load(self.audio_path, sr=sr)
        f0_series = []

        frame_len = 1024
        #Hanning frames helps to reduce aliasing when creating discrete sections
        window = np.hanning(frame_len)

        for i in range(0, len(y) - frame_len, hop_length):
            # Creates the frames for the Fourier Transform
            frame = y[i:i+frame_len] * window
            spectrum = np.fft.rfft(frame) # Applies a Discrete Fourier Transform to compenent frequencies
            magnitude = np.abs(spectrum) # Uses only the positive frequencies
            hps = magnitude.copy() 

            for h in range(2, harmonics + 1):
                dec = magnitude[::h] #Compresses the frequency by a factor of 2 then 3 then 4 so that the frequencies align with the fundamental.
                hps[:len(dec)] *= dec

            freqs = np.fft.rfftfreq(len(frame), d=1/sr)
            pitch_idx = np.argmax(hps) #Find the peaks in the newly calculated harmonic spectrum
            f0 = freqs[pitch_idx]
            f0_series.append(f0 if 50 < f0 < 1000 else 0)  # Sanity check

        return f0_series
    
    def pitch_to_note(self, f0_series=None):
        #If the pitches were not already observed from the audio then it will capture realtime
        if f0_series is None:
            f0_series = self.detect_pitch_hps()
        #Determines the note name for every frequency detected
        note_series = [
            librosa.hz_to_note(f) if f > 0 else "-"
            for f in f0_series
        ]
        return note_series
