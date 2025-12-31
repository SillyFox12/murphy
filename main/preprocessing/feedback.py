from typing import Any
import numpy as np
import joblib
import pandas as pd
import json as js
from collections import Counter

class NoteFeedback:
    def __init__(self):
        pass

    def load(self, file):
        with open(file, 'r') as f:
            correct = js.load(f)
        sequence_key = next(iter(correct))
        correct = np.array([note["note"] for note in correct[sequence_key]])
        return correct
    
    def check_notes(self, correct, played):
        is_correct = []
        for i in range(len(correct)):
            if correct[i] == played[i]:
                is_correct.append(True)
                print(f"Note {i+1}: Correct")
            else:
                is_correct.append(False)
                print(f"Note {i+1}: Incorrect - Expected {correct[i]}, Got {played[i]}")
        return is_correct
    
class NoteEventStream:
    def __init__(self, audio_csv: str):
        self.df = pd.read_csv(audio_csv)

    def iter_events(self):
        for _, row in self.df.iterrows():
            yield {
                "onset": row["onset"],
                "duration": row["duration"],
                "note": row["note"],
                "frequency": row["frequency"],
                "cents": row["cents"],
            }

class FrameWindowExtractor:
    def __init__(self, fps: float, window_ms: int = 120):
        self.fps = fps
        self.half_window_frames = int((window_ms / 1000) * fps / 2)

    def get_window_indices(self, onset_time: float, n_frames: int):
        center = int(onset_time * self.fps)
        start = max(center - self.half_window_frames, 0)
        end = min(center + self.half_window_frames, n_frames - 1)
        return start, end

class TechniqueModel:
    def __init__(self, model_path: str, threshold: float = 0.6):
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def predict_window(self, X_window: pd.DataFrame):
        """
        Returns:
            vote (0/1), confidence (mean prob)
        """
        probs = self.model.predict_proba(X_window)[:, 1]
        confidence = probs.mean()
        vote = int(confidence >= self.threshold)
        return vote, confidence

class MajorityVote:
    def __init__(self, min_ratio: float = 0.6):
        self.min_ratio = min_ratio

    def decide(self, votes: list[int]):
        if len(votes) == 0:
            return 0, 0.0

        count = Counter(votes)
        dominant = count.most_common(1)[0]
        ratio = dominant[1] / len(votes)

        if dominant[0] == 1 and ratio >= self.min_ratio:
            return 1, ratio
        
        return 0, ratio

class EventFeedbackEngine:
    def __init__(
        self,
        features_csv: str,
        audio_csv: str,
        fps: float,
        models: dict
    ):
        self.features = pd.read_csv(features_csv)
        self.audio = pd.read_csv(audio_csv)
        self.n_frames = len(self.features)

        self.window = FrameWindowExtractor(fps)
        self.voter = MajorityVote()
        self.models = models

    def evaluate(self):
        results = []

        for _, note_event in self.audio.iterrows():
            onset = note_event["onset"]
            note = note_event["note"]

            start, end = self.window.get_window_indices(onset, self.n_frames)
            frame_slice = self.features.iloc[start:end]

            feedback = {
                "onset": onset,
                "note": note,
            }

            for name, model in self.models.items():
                frame_votes = []
                confidences = []

                for _, frame in frame_slice.iterrows():
                    X = frame.to_frame().T
                    vote, conf = model.predict_window(X)
                    frame_votes.append(vote)
                    confidences.append(conf)

                decision, ratio = self.voter.decide(frame_votes)

                feedback[f"{name}_error"] = decision
                feedback[f"{name}_confidence"] = np.mean(confidences)

            results.append(feedback)

        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test Note Centric Feedback Engine
    FPS = 30.0

    models = {
        "wrist": TechniqueModel("models/wrist_error_detector.pkl", threshold=0.6),
        "finger": TechniqueModel("models/finger_error_detector.pkl", threshold=0.6),
        "thumb": TechniqueModel("models/thumb_error_detector.pkl", threshold=0.6),
    }

    engine = EventFeedbackEngine(
        features_csv="data/features.csv",
        audio_csv="data/audio.csv",
        fps=FPS,
        models=models
    )

    df_feedback = engine.evaluate()
    print(df_feedback)

    df_feedback.to_csv("data/test_feedback_output.csv", index=False)

