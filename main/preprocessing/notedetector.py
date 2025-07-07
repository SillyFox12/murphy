# -*- coding: utf-8 -*-
# Necessary Dependencies
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from typing import List, Dict, Any, Optional
import csv

# --- Configuration ---

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
        self.frame_analysis_interval = frame_analysis_interval
        self.audio_sr = audio_sr
        self.audio_hop_length = audio_hop_length
        self.audio_frame_size = audio_frame_size
        self.audio_fmin = audio_fmin
        self.audio_fmax = audio_fmax
        
        # --- Aliases for librosa compatibility ---
        # This simplifies calls to librosa functions.
        self.sr = self.audio_sr
        self.hop_length = self.audio_hop_length
        self.frame_size = self.audio_frame_size
        self.fmin = self.audio_fmin
        self.fmax = self.audio_fmax

# --- Chord Recognition Module ---

class ChordRecognizer:
    """Recognizes chords from chroma features using template matching."""
    NOTE_NAMES: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    CHORD_TEMPLATES: Dict[str, np.ndarray] = {
        'N.C.': np.array([0]*12),
        # Major Chords
        'C':    np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
        'C#':   np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 1),
        'D':    np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 2),
        'D#':   np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 3),
        'E':    np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 4),
        'F':    np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 5),
        'F#':   np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 6),
        'G':    np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 7),
        'G#':   np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 8),
        'A':    np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 9),
        'A#':   np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 10),
        'B':    np.roll(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 11),
        # Minor Chords
        'Cm':   np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
        'C#m':  np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 1),
        'Dm':   np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 2),
        'D#m':  np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 3),
        'Em':   np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 4),
        'Fm':   np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 5),
        'F#m':  np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 6),
        'Gm':   np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 7),
        'G#m':  np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 8),
        'Am':   np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 9),
        'A#m':  np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 10),
        'Bm':   np.roll(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), 11),
    }
    
    def __init__(self, chroma_threshold: float = 0.85):
        self.chroma_threshold = chroma_threshold

    def find_best_match(self, chroma_vector: np.ndarray) -> str:
        if not np.any(chroma_vector):
            return 'N.C.'

        best_match = 'N.C.'
        max_similarity = -1.0
        
        for name, template in self.CHORD_TEMPLATES.items():
            similarity = 1 - cosine(chroma_vector, template)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name
        
        return best_match if max_similarity >= self.chroma_threshold else 'N.C.'

# --- Main Analyzer ---

class AudioAnalyzer:
    """Audio analyzer for pitch and chord detection."""
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initializes the analyzer.
        
        Args:
            config: A configuration object. If None, default settings are used.
        """
        self.config = config if config else AnalysisConfig()
        self.chord_recognizer = ChordRecognizer()

    def analyze_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """Analyzes an audio file to identify a time-series of pitches and chords."""
        try:
            # All parameters are now correctly sourced from self.config
            y, _ = librosa.load(audio_path, sr=self.config.sr)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return []

        stft_matrix = np.abs(librosa.stft(y, 
                                          n_fft=self.config.frame_size, 
                                          hop_length=self.config.hop_length))
                                          
        chromagram = librosa.feature.chroma_stft(S=stft_matrix, 
                                                 sr=self.config.sr,
                                                 n_fft=self.config.frame_size,
                                                 hop_length=self.config.hop_length)

        pitches, voiced_flags, _ = librosa.pyin(y,
                                                fmin=self.config.fmin,
                                                fmax=self.config.fmax,
                                                sr=self.config.sr,
                                                frame_length=self.config.frame_size,
                                                hop_length=self.config.hop_length,
                                                fill_na=None)

        times = librosa.times_like(pitches, sr=self.config.sr, hop_length=self.config.hop_length)
        results = []

        for i, time in enumerate(times):
            pitch_hz = pitches[i]
            is_voiced = voiced_flags[i]
            
            chroma_frame = chromagram[:, i]
            is_polyphonic = np.sum(chroma_frame > 0.4) > 2

            event = {"time_sec": round(time, 2), "type": "silence", "value": None}

            if is_voiced:
                if is_polyphonic:
                    chord = self.chord_recognizer.find_best_match(chroma_frame)
                    if chord != 'N.C.':
                        event["type"] = "chord"
                        event["value"] = chord
                else:
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

        consolidated.append({
            "start_time_sec": current_event['time_sec'],
            "end_time_sec": results[-1]['time_sec'] + 0.1,
            "type": current_event['type'],
            "value": current_event['value']
        })
        
        return [res for res in consolidated if res['type'] != 'silence']
    
    @staticmethod
    def export_to_csv(results: List[Dict[str, Any]], output_path: str = "analysis_results.csv") -> None:
        """Exports pitch/chord analysis results to a CSV file."""
        if not results:
            print("No results to export.")
            return

        fieldnames = ["start_time_sec", "end_time_sec", "type", "value"]

        try:
            with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    filtered_row = {key: row.get(key) for key in fieldnames}
                    writer.writerow(filtered_row)
            print(f"[✅] CSV export complete: {output_path}")
        except Exception as e:
            print(f"[❌] Failed to write CSV: {e}")