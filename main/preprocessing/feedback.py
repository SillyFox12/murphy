import numpy as np
import json as js

class FeedbackPreprocessor:
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

if __name__ == "__main__":
    preprocessor = FeedbackPreprocessor()
    correct_notes = preprocessor.load("data/lesson/c_maj.json")
    played_notes = ["A3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4", "B3", "A3", "G3", "F3", "E3", "D3", "C3"]
    results = preprocessor.check_notes(correct_notes, played_notes)
    print(f"Feedback Results: {results}")
