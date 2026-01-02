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
    def __init__(self, model_path: str):

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["features"]

        # Ensure the model is a binary classifier (most scikit-learn classifiers have .classes_)
        if not (hasattr(self.model, "classes_") and len(self.model.classes_) == 2):
            raise ValueError("The loaded model must be a binary scikit-learn classifier with exactly two classes.")
        
        # The positive class is always at index 1 (scikit-learn sorts classes_ and aligns probabilities/decisions to it)
        self.positive_class = self.model.classes_[1]

    def predict_frame(self, X: pd.DataFrame):
        """
        Returns:
            vote (0/1), confidence ∈ [0,1]
        """
        if len(X) != 1:
            raise ValueError("predict_frame expects a DataFrame with exactly one row (a single frame).")

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            conf = float(probs[0, 1])  # Probability of the positive class
            vote = 1 if conf >= 0.5 else 0
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(X)[0])
            conf = 1 / (1 + np.exp(-score))  # Sigmoid to approximate probability
            vote = 1 if conf >= 0.5 else 0
        elif hasattr(self.model, "predict"):
            pred = self.model.predict(X)[0]
            vote = 1 if pred == self.positive_class else 0
            conf = 1.0
        else:
            raise AttributeError("The loaded model has no supported prediction method.")

        return vote, conf
    
    def predict_batch(self, X_window: pd.DataFrame):
        X_feats = X_window[self.feature_cols]

        probs = self.model.predict_proba(X_feats)

        # Binary classifier assumed
        votes = (probs[:, 1] > 0.5).astype(int)
        confs = probs[:, 1]

        return votes, confs


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
    
    def decide_weighted(self, votes: np.ndarray, weights: np.ndarray):
        """
        votes: 0/1 predictions
        weights: same length, ≥0
        """
        if len(votes) == 0:
            return 0, 0.0

        weighted_sum = np.sum(votes * weights)
        weight_total = np.sum(weights)

        ratio = weighted_sum / weight_total if weight_total > 0 else 0.0
        decision = int(ratio >= self.min_ratio)

        return decision, ratio

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

    def evaluate(self) -> pd.DataFrame:
        results = []

        feature_cols = [
            c for c in self.features.columns
            if c not in ("frame", "hand_index")
        ]

        for _, note_event in self.audio.iterrows():
            onset = note_event["onset"]
            note = note_event["note"]

            # 1️⃣ Extract temporal window
            start, end = self.window.get_window_indices(onset, self.n_frames)
            frame_slice = self.features.iloc[start:end + 1]

            if frame_slice.empty:
                continue

            # 2️⃣ Compute temporal weights (per note)
            frame_indices = np.arange(start, end + 1)
            center = (start + end) / 2
            dist = np.abs(frame_indices - center)

            sigma = max(len(frame_indices) / 4, 1e-6)
            weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))

            feedback = {
                "onset": onset,
                "note": note,
            }

            # 3️⃣ Model evaluation
            for name, model in self.models.items():
                X_window = frame_slice[feature_cols]

                votes, confs = model.predict_batch(X_window)
                decision, ratio = self.voter.decide_weighted(
                    votes=votes,
                    weights=weights
                )

                feedback[f"{name}_error"] = decision
                feedback[f"{name}_confidence"] = ratio

            results.append(feedback)

        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test Note Centric Feedback Engine
    FPS = 30.0

    models = {
        "wrist": TechniqueModel("models/wrist_error_detector.pkl"),
        "finger": TechniqueModel("models/finger_error_detector.pkl"),
        "thumb": TechniqueModel("models/thumb_error_detector.pkl"),
    }

    engine = EventFeedbackEngine(
        features_csv="data/test/visual_data/processed/hand_features.csv",
        audio_csv="data/test/audio_data/processed/slanted_finger.csv",
        fps=FPS,
        models=models
    )

    df_feedback = engine.evaluate()
    print(df_feedback)

    df_feedback.to_csv("data/test_feedback_output.csv", index=False)

