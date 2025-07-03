from unittest import result
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from typing import List, Dict, Any, Tuple, Optional, Set
import csv

# --- Configuration ---

class AnalysisConfig:
    """Encapsulates all analysis parameters for clarity and ease of use."""
    def __init__(self, 
                 sr: int = 22050, 
                 frame_size: int = 4096, 
                 hop_length: int = 1024,
                 fmin: float = librosa.note_to_hz('C2'),
                 fmax: float = librosa.note_to_hz('C7')):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length
        # Frequency range for pYIN pitch detection
        self.fmin = fmin
        self.fmax = fmax

# --- Chord Recognition Module ---

class ChordRecognizer:
    """
    Recognizes chords from chroma features using template matching.
    """
    NOTE_NAMES: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Scalable and clear chord templates. Represented as binary chroma vectors.
    CHORD_TEMPLATES: Dict[str, np.ndarray] = {
        'N.C.': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # No Chord
        # Major Chords (Root, Major Third, Perfect Fifth)
        'C':    np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
        'C#':   np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
        'D':    np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
        'D#':   np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
        'E':    np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
        'F':    np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
        'F#':   np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]),
        'G':    np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]),
        'G#':   np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]),
        'A':    np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
        'A#':   np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
        'B':    np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]),
        # Minor Chords (Root, Minor Third, Perfect Fifth)
        'Cm':   np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
        'C#m':  np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
        'Dm':   np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
        'D#m':  np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
        'Em':   np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
        'Fm':   np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
        'F#m':  np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
        'Gm':   np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]),
        'G#m':  np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]),
        'Am':   np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
        'A#m':  np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
        'Bm':   np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    }
    
    def __init__(self, chroma_threshold: float = 0.85):
        #Initializes the chord recognizer
        self.chroma_threshold = chroma_threshold

    def find_best_match(self, chroma_vector: np.ndarray) -> str:
        """
        Finds the best matching chord for a given chroma vector.

        Args:
            chroma_vector: A 12-element array representing musical note energy.

        Returns:
            The name of the best-matched chord or 'N.C.' (No Chord).
        """
        if not np.any(chroma_vector):
             return 'N.C.'

        best_match = 'N.C.'
        max_similarity = -1.0
        
        # We use cosine similarity, which is robust to volume changes
        for name, template in self.CHORD_TEMPLATES.items():
            similarity = 1 - cosine(chroma_vector, template)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name
        
        return best_match if max_similarity >= self.chroma_threshold else 'N.C.'

# --- Main Analyzer ---

class AudioAnalyzer:
    """
    A robust and efficient audio analyzer for pitch and chord detection.
    """
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
        self.chord_recognizer = ChordRecognizer()

    def analyze_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes an audio file to identify a time-series of pitches and chords.

        Args:
            audio_path: The path to the input audio file.

        Returns:
            A list of dictionaries, each describing the event (pitch or chord)
            and its timing within the audio file.
        """
        try:
            y, _ = librosa.load(audio_path, sr=self.config.sr)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return []

        # 1. COMPUTE ONCE: Perform STFT and get magnitude and chroma features
        stft_matrix = np.abs(librosa.stft(y, 
                                          n_fft=self.config.frame_size, 
                                          hop_length=self.config.hop_length))
                                          
        chromagram = librosa.feature.chroma_stft(S=stft_matrix, 
                                                 sr=self.config.sr,
                                                 n_fft=self.config.frame_size,
                                                 hop_length=self.config.hop_length)

        # 2. PITCH DETECTION: Use (pYIN) for pitches
        pitches, voiced_flags, _ = librosa.pyin(y,
                                                fmin=self.config.fmin,
                                                fmax=self.config.fmax,
                                                sr=self.config.sr,
                                                frame_length=self.config.frame_size,
                                                hop_length=self.config.hop_length,
                                                fill_na=None) # Keep NaN for unvoiced frames

        times = librosa.times_like(pitches, sr=self.config.sr, hop_length=self.config.hop_length)
        results = []

        # 3. DECISION LOGIC: Iterate through time frames and decide
        for i, time in enumerate(times):
            pitch_hz = pitches[i]
            is_voiced = voiced_flags[i]
            
            # Use chroma features to determine polyphony
            chroma_frame = chromagram[:, i]
            # A simple but more robust heuristic: if more than 2 chroma bins are strong,
            # it's likely a chord.
            is_polyphonic = np.sum(chroma_frame > 0.4) > 2

            event = {
                "time_sec": round(time, 2),
                "type": "silence",
                "value": None
            }

            if is_voiced:
                if is_polyphonic:
                    chord = self.chord_recognizer.find_best_match(chroma_frame)
                    if chord != 'N.C.':
                        event["type"] = "chord"
                        event["value"] = chord
                else:
                    # It's a single, voiced pitch
                    event["type"] = "pitch"
                    event["value"] = librosa.hz_to_note(pitch_hz)
            
            results.append(event)
            
        return self._consolidate_results(results)

    @staticmethod
    def _consolidate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges consecutive identical events for a cleaner output."""
        if not results:
            return []
            
        consolidated = []
        current_event = results[0]
        
        for i in range(1, len(results)):
            if results[i]['value'] != current_event['value']:
                consolidated.append({
                    "start_time_sec": current_event['time_sec'],
                    "end_time_sec": results[i]['time_sec'],
                    "type": current_event['type'],
                    "value": current_event['value']
                })
                current_event = results[i]

        # Add the last event
        consolidated.append({
            "start_time_sec": current_event['time_sec'],
            "end_time_sec": results[-1]['time_sec'] + 0.1, # Add small buffer to end time
            "type": current_event['type'],
            "value": current_event['value']
        })
        
        return [res for res in consolidated if res['type'] != 'silence']
    
    @staticmethod
    def export_to_csv(results: List[Dict[str, Any]], output_path: str = "analysis_results.csv") -> None:
        """
        Exports pitch/chord analysis results to a CSV file.

        Args:
            results: List of dictionaries with analysis results.
            output_path: Path to the CSV file to be written.
        """
        if not results:
            print("No results to export.")
            return

        # Determine column keys from first result
        fieldnames = ["start_time_sec", "end_time_sec", "type", "value"]

        try:
            with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in results:
                    # Ensure missing keys don't crash export
                    filtered_row = {key: row.get(key, None) for key in fieldnames}
                    writer.writerow(filtered_row)

            print(f"[✅] CSV export complete: {output_path}")
        except Exception as e:
            print(f"[❌] Failed to write CSV: {e}")


if __name__ == '__main__': 
    # Instantiate the analyzer with default config and run it
    analyzer = AudioAnalyzer()
    analysis_results = analyzer.analyze_audio("./data/test.wav")
    
    # Print the results in a readable format
    # Export to CSV
    AudioAnalyzer.export_to_csv(analysis_results, output_path="./data/pitch_chord_analysis.csv")
    
    # Expected output should show a 'C' chord followed by a 'G' pitch.