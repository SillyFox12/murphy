import cv2
import matplotlib.pyplot as plt
import numpy as np

class FretboardDebugVisualizer:
    """A dedicated class to generate a visual debugging panel for the detector."""
    def __init__(self, detector: None):
        self.detector = detector

    def generate_debug_panel(self):
        """Creates and displays a multi-panel plot of the detection stages."""
        if not self.detector.debug_data:
            print("No debug data found. Run detector.detect_fretboard() first.")
            return

        data = self.detector.debug_data
        fig, axes = plt.subplots(2, 3, figsize=(24, 13))
        fig.suptitle("Fretboard Detector - Debugging Panel", fontsize=20)
        ax = axes.ravel()

        def draw_lines(img, lines, color=(0,0,255)):
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), color, 2)
            return img

        # Panel 1: Original Image
        ax[0].imshow(cv2.cvtColor(data['original'], cv2.COLOR_BGR2RGB)); ax[0].set_title("1. Original Image"); ax[0].axis('off')

        # Panel 2: Preprocessed Image
        ax[1].imshow(data['preprocessed'], cmap='gray'); ax[1].set_title("2. Preprocessed"); ax[1].axis('off')

        # Panel 3: Canny Edges
        ax[2].imshow(data['edges'], cmap='gray'); ax[2].set_title("3. Canny Edges"); ax[2].axis('off')
        
        # Panel 4: Raw Hough Lines
        ax[3].imshow(cv2.cvtColor(draw_lines(data['original'].copy(), data.get('raw_lines')), cv2.COLOR_BGR2RGB)); ax[3].set_title("4. All Raw Lines Found"); ax[3].axis('off')
        
        # Panel 5: Grid-Filtered Lines
        grid_lines_img = data['original'].copy()
        draw_lines(grid_lines_img, data.get('grid_filtered_frets', []), (255, 255, 255)) # White Frets
        draw_lines(grid_lines_img, data.get('grid_filtered_strings', []), (255, 0, 255)) # Magenta Strings
        ax[4].imshow(cv2.cvtColor(grid_lines_img, cv2.COLOR_BGR2RGB)); ax[4].set_title("5. Grid-Filtered Lines"); ax[4].axis('off')

        # Panel 6: Final Result
        final_img = data['original'].copy()
        if self.detector.is_valid:
            for i, l in enumerate(self.detector.frets): cv2.line(final_img, (l[0], l[1]), (l[2], l[3]), (0, 255 - i*15, i*15), 2)
            for i, l in enumerate(self.detector.strings): cv2.line(final_img, (l[0], l[1]), (l[2], l[3]), (255, i*30, 0), 2)
        
        conf_text = f"Confidence: {self.detector.detection_confidence:.2f}"
        color = (0, 255, 0) if self.detector.is_valid else (0, 0, 255)
        cv2.putText(final_img, conf_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        ax[5].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)); ax[5].set_title("6. Final Validated Grid"); ax[5].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
