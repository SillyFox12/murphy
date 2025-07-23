# fretboard_geometry.py
import numpy as np
from typing import List
from sklearn.decomposition import PCA

class FretboardValidator:
    """A dedicated, testable module for validating fretboard geometry."""
    FRET_SPACING_RATIO_IDEAL = 1 / (2**(1/12))  # ~0.94387

    @staticmethod
    def validate_fret_spacing(frets: List[np.ndarray]) -> float:
        """
        Calculates a score from 0.0 to 1.0 based on the ideal exponential
        fret spacing rule (12-Tone Equal Temperament). A score closer to 1.0
        indicates a better match.
        """
        if len(frets) < 3:
            return 0.0

        # Calculates the midpoints of each detected fret
        midpoints = np.array([[(l[0] + l[2]) / 2, (l[1] + l[3]) / 2] for l in frets])
        # Finds the main axis of fret alignment
        pca = PCA(n_components=1).fit(midpoints)
        # Creates a number line with the midpoint positions projected onto it.
        projections = midpoints @ pca.components_.T
        # Sorts the projections to analyze spacing
        sorted_indices = np.argsort(projections[:, 0])
        # Analyzes the spacing between consecutive frets
        spacings = np.diff(projections[sorted_indices, 0])
        # Fails gracefully if not enough spacing are available for ratio calculation
        if len(spacings) < 2:
            return 0.0
        # Calculates the ratio of consecutive spacings and compares to ideal ratio
        ratios = spacings[1:] / spacings[:-1]
        errors = np.abs(ratios - FretboardValidator.FRET_SPACING_RATIO_IDEAL) / FretboardValidator.FRET_SPACING_RATIO_IDEAL
        avg_error = np.mean(errors)
        
        return np.clip(1.0 - avg_error * 2.0, 0, 1)

    @staticmethod
    def validate_string_spacing(strings: List[np.ndarray]) -> float:
        """
        Calculates a score from 0.0 to 1.0 based on the consistency of
        string spacing. A score closer to 1.0 indicates more evenly
        spaced strings.
        """
        if len(strings) < 3:
            return 0.0
        
        # Calculates the midpoints of each detected string
        midpoints = np.array([[(l[0] + l[2]) / 2, (l[1] + l[3]) / 2] for l in strings])
        # Projects the midpoints onto the perpendicular axis of the string alignment in order to analyze spacing
        pca = PCA(n_components=2).fit(midpoints)
        perp_axis = pca.components_[1]
        projections = midpoints @ perp_axis.T
        projections.sort()

        # Calculates the spacing between consecutive strings
        spacings = np.diff(projections)
        if np.mean(spacings) == 0: return 0.0
        
        # Calculates the consistency of the string spacing
        consistency = np.std(spacings) / np.mean(spacings)
        return np.clip(1.0 - consistency * 3.0, 0, 1)