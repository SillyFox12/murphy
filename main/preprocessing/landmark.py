import csv
import os

class LandmarkLogger:
    def __init__(self, output_dir, filename="landmarks.csv"):
        self.filepath = os.path.join(output_dir, filename)
        self.header_written = False

    def save_landmarks(self, frame_number, results):
        if not results.multi_hand_landmarks:
            return  # Skip frames with no hands

        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                row = [frame_number, hand_idx]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)

    def write_header(self):
        if not self.header_written and not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["frame", "hand"]
                for i in range(21):
                    header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
                writer.writerow(header)
            self.header_written = True
