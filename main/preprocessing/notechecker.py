import pandas as pd
import librosa

def check_ordered_notes_pandas(csv_path, target_notes, tolerance_semitones=0.5):
    """Checks whether the nth detected note matches the nth target note using pandas."""
    
    # === Step 1: Load the CSV ===
    df = pd.read_csv(csv_path)

    # Filter for valid pitch rows
    df = df[df["type"] == "pitch"].dropna(subset=["value"]).reset_index(drop=True)
    
    # === Step 2: Handle length mismatches ===
    n_targets = len(target_notes)
    n_detected = len(df)
    max_len = max(n_targets, n_detected)

    # Extend lists to equal length (fill missing values with None)
    padded_targets = target_notes + [None] * (max_len - n_targets)
    if n_detected < max_len:
        extra_rows = pd.DataFrame({
            "start_time_sec": [None] * (max_len - n_detected),
            "end_time_sec": [None] * (max_len - n_detected),
            "value": [None] * (max_len - n_detected)
        })
        df = pd.concat([df, extra_rows], ignore_index=True)

    # === Step 3: Add target notes and compare ===
    df["target_note"] = padded_targets

    def check_correct(row):
        """Compare played vs target note using semitone tolerance."""
        if pd.isna(row["target_note"]) or pd.isna(row["value"]):
            return False
        diff = abs(librosa.note_to_midi(row["target_note"]) - librosa.note_to_midi(row["value"]))
        return diff <= tolerance_semitones

    df["is_correct"] = df.apply(check_correct, axis=1)

    # === Step 4: Display summary ===
    print(f"\n=== Ordered Note Check (pandas) ===")
    for i, row in df.iterrows():
        target = row['target_note']
        played = row['value']
        status = "✅ Correct" if row['is_correct'] else "❌ Incorrect"
        print(f"{status}: Target={target} | Played={played} | Time={row['start_time_sec']}s")

    accuracy = df["is_correct"].mean() * 100
    print(f"\n🎯 Accuracy: {df['is_correct'].sum()}/{len(df)} = {accuracy:.1f}%")

    return df


# Example usage
if __name__ == "__main__":
    target_sequence = ["A4"]  # expected notes in order
    csv_path = "analysis_results.csv"
    results_df = check_ordered_notes_pandas(csv_path, target_sequence)

    # Optionally, save to a new CSV
    results_df.to_csv("checked_results_pandas.csv", index=False)
