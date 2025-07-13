# -*- coding: utf-8 -*-
"""
A professional-grade 3D hand gesture visualizer.

This script reconstructs and animates a 3D hand skeleton from the engineered
feature data (angles and distances), providing a true representation of the
captured gesture for data validation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple

class HandGestureVisualizer:
    """
    Reconstructs and visualizes 3D hand gestures from a feature CSV file.
    """
    # Define canonical properties of the hand skeleton.
    # Segment lengths are ratios, providing a more realistic look.
    SEGMENT_LENGTHS = {"MCP_PIP": 0.8, "PIP_DIP": 0.6, "DIP_TIP": 0.5}
    FINGER_NAMES = ["INDEX", "MIDDLE", "RING", "PINKY", "THUMB"]

    def __init__(self, csv_path: str):
        """
        Initializes the visualizer with the path to the hand features CSV.

        Args:
            csv_path (str): The full path to the 'hand_features.csv' file.
        """
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Successfully loaded {len(self.df)} frames from {csv_path}")
        except FileNotFoundError:
            print(f"Error: The file was not found at {csv_path}")
            raise
        
        # Store the plot components
        self.fig, self.ax, self.lines = self._setup_plot()

    def _setup_plot(self) -> Tuple[plt.figure, plt.axes, Dict[str, plt.Line2D]]:
        """Creates the 3D plot and line objects for the animation."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Hand Gesture Reconstruction")
        
        # Set realistic plot limits and labels
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        
        # Invert Y-axis for a more intuitive "upright" hand view
        ax.invert_yaxis()

        # Create a line object for each finger
        lines = {finger: ax.plot([], [], [], 'o-', lw=4, markersize=8)[0] for finger in self.FINGER_NAMES}
        return fig, ax, lines

    def _reconstruct_mcp_positions(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """
        Calculates the 3D positions of the MCP (knuckle) joints.
        This uses the inter-finger distances to create a realistic hand spread.
        """
        mcp_positions = {}

        # A simple but effective heuristic:
        # 1. Place the Middle finger's MCP at the origin.
        # 2. Place the Index and Ring fingers based on their distance to the Middle.
        # 3. Place the Pinky based on its distance to the Ring.
        
        mcp_positions["MIDDLE"] = np.array([0, 0, 0])
        
        # Use a small vertical offset to create a natural knuckle arch
        dist_index_middle = row.get('dist_index_middle', 0.8)
        mcp_positions["INDEX"] = np.array([-dist_index_middle, 0, 0.1])

        dist_middle_ring = row.get('dist_middle_ring', 0.8)
        mcp_positions["RING"] = np.array([dist_middle_ring, 0, 0.1])
        
        dist_ring_pinky = row.get('dist_ring_pinky', 0.7)
        mcp_positions["PINKY"] = mcp_positions["RING"] + np.array([dist_ring_pinky, 0, -0.1])
        
        # Place thumb based on its distance to the index MCP
        dist_thumb_index = row.get('dist_thumb_index', 1.5)
        mcp_positions["THUMB"] = np.array([-dist_thumb_index * 0.7, -dist_thumb_index * 0.7, 0])
        
        return mcp_positions

    def _reconstruct_finger(self, mcp_pos: np.ndarray, pip_angle_deg: float, dip_angle_deg: float, base_direction: np.ndarray) -> np.ndarray:
        """
        Calculates the 3D coordinates of a single finger using forward kinematics.
        """
        # Convert angles for kinematic calculations
        pip_angle_rad = np.radians(180 - pip_angle_deg)
        dip_angle_rad = np.radians(180 - dip_angle_deg)
        
        # Define rotation axis (perpendicular to the base direction and Z-axis)
        rot_axis = np.cross(base_direction, np.array([0, 0, 1]))
        rot_axis /= np.linalg.norm(rot_axis)

        # Function to apply rotation
        def rotate(vector, axis, angle_rad):
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            return vector * cos_a + np.cross(axis, vector) * sin_a + axis * np.dot(axis, vector) * (1 - cos_a)

        # Calculate joint positions
        pip_pos = mcp_pos + base_direction * self.SEGMENT_LENGTHS["MCP_PIP"]
        
        vec_pip_dip = rotate(pip_pos - mcp_pos, rot_axis, pip_angle_rad)
        dip_pos = pip_pos + vec_pip_dip / np.linalg.norm(vec_pip_dip) * self.SEGMENT_LENGTHS["PIP_DIP"]
        
        vec_dip_tip = rotate(dip_pos - pip_pos, rot_axis, dip_angle_rad)
        tip_pos = dip_pos + vec_dip_tip / np.linalg.norm(vec_dip_tip) * self.SEGMENT_LENGTHS["DIP_TIP"]
        
        return np.array([mcp_pos, pip_pos, dip_pos, tip_pos])

    def _update_frame(self, frame_idx: int) -> List[plt.Line2D]:
        """The core animation function, called for each frame."""
        row = self.df.iloc[frame_idx]
        
        # 1. Reconstruct the base of the hand (the knuckles)
        mcp_positions = self._reconstruct_mcp_positions(row)

        # 2. Reconstruct each finger based on its knuckle position and angles
        for finger_name in self.FINGER_NAMES:
            name_lower = finger_name.lower()
            pip_angle = row.get(f'angle_{name_lower}_pip', 180)
            dip_angle = row.get(f'angle_{name_lower}_dip', 180)
            
            mcp_pos = mcp_positions[finger_name]
            
            # Define a base direction for each finger
            if finger_name == "THUMB":
                base_dir = np.array([0.5, -0.5, 0.1])
            else:
                base_dir = np.array([0, 0.1, 1]) # Fingers point generally "up"
            base_dir /= np.linalg.norm(base_dir)

            coords = self._reconstruct_finger(mcp_pos, pip_angle, dip_angle, base_dir)
            
            # Update the plot data
            self.lines[finger_name].set_data(coords[:, 0], coords[:, 1])
            self.lines[finger_name].set_3d_properties(coords[:, 2])

        self.ax.set_title(f"3D Hand Gesture Reconstruction (Frame {frame_idx})")
        return list(self.lines.values())

    def run_animation(self):
        """Starts and displays the 3D animation."""
        # Create the animation object
        ani = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.df),
            interval=50,  # Milliseconds between frames
            blit=False  # Blit is often problematic in 3D
        )
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 2. Instantiate the visualizer with the path.
    try:
        visualizer = HandGestureVisualizer(csv_path="./data/session4/hand_features.csv")
        print("Hand Gesture Visualizer initialized successfully.")
        # 3. Run the animation.
        visualizer.run_animation()
    except FileNotFoundError:
        print("\nDemonstration failed. Could not find the CSV file.")