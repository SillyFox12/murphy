import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple
import os
import csv

class HandGestureVisualizer:
    """
    Reconstructs and visualizes 3D hand gestures from a feature CSV file.
    """
    SEGMENT_LENGTHS = {"MCP_PIP": 1.0, "PIP_DIP": 0.7, "DIP_TIP": 0.6}
    FINGER_NAMES = ["INDEX", "MIDDLE", "RING", "PINKY", "THUMB"]

    def __init__(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Successfully loaded {len(self.df)} frames from {csv_path}")
        except FileNotFoundError:
            print(f"Error: The file was not found at {csv_path}")
            raise
        
        self.fig, self.ax, self.lines = self._setup_plot()

    def _setup_plot(self) -> Tuple[plt.Figure, plt.Axes, Dict[str, plt.Line2D]]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Hand Gesture Reconstruction")
        
        ax.set_xlim([-3, 3]); ax.set_ylim([-3, 3]); ax.set_zlim([-3, 3])
        ax.set_xlabel("X Axis"); ax.set_ylabel("Y Axis"); ax.set_zlabel("Z Axis")
        
        ax.invert_yaxis()
        ax.view_init(elev=90., azim=-90)

        lines = {finger: ax.plot([], [], [], 'o-', lw=4, markersize=8)[0] for finger in self.FINGER_NAMES}
        return fig, ax, lines

    def _reconstruct_mcp_positions(self, row: pd.Series) -> Dict[str, np.ndarray]:
        mcp_positions = {}
        mcp_positions["MIDDLE"] = np.array([0.0, 0.0, 0.0])
        
        dist_index_middle = row.get('dist_index_middle', 0.8)
        mcp_positions["INDEX"] = np.array([-dist_index_middle, 0.0, 0.1])

        dist_middle_ring = row.get('dist_middle_ring', 0.8)
        mcp_positions["RING"] = np.array([dist_middle_ring, 0.0, 0.1])
        
        dist_ring_pinky = row.get('dist_ring_pinky', 0.7)
        mcp_positions["PINKY"] = mcp_positions["RING"] + np.array([dist_ring_pinky, 0.0, -0.1])
        
        dist_thumb_index = row.get('dist_thumb_index', 1.5)
        mcp_positions["THUMB"] = np.array([-dist_thumb_index * 0.7, -dist_thumb_index * 0.7, 0.0])
        
        return mcp_positions

    def _reconstruct_finger(self, mcp_pos: np.ndarray, pip_angle_deg: float, dip_angle_deg: float, base_direction: np.ndarray, up_vector: np.ndarray) -> np.ndarray:
        # Determines finger curl.
        pip_angle_rad = np.radians(180 - pip_angle_deg)
        dip_angle_rad = np.radians(180 - dip_angle_deg)
        
        rot_axis = np.cross(base_direction, up_vector)
        if np.linalg.norm(rot_axis) != 0:
            rot_axis /= np.linalg.norm(rot_axis)

        def rotate(vector, axis, angle_rad):
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            return vector * cos_a + np.cross(axis, vector) * sin_a + axis * np.dot(axis, vector) * (1 - cos_a)

        pip_pos = mcp_pos + base_direction * self.SEGMENT_LENGTHS["MCP_PIP"]
        
        vec_pip_dip = rotate(pip_pos - mcp_pos, rot_axis, pip_angle_rad)
        if np.linalg.norm(vec_pip_dip) != 0:
             vec_pip_dip /= np.linalg.norm(vec_pip_dip)
        dip_pos = pip_pos + vec_pip_dip * self.SEGMENT_LENGTHS["PIP_DIP"]
        
        vec_dip_tip = rotate(dip_pos - pip_pos, rot_axis, dip_angle_rad)
        if np.linalg.norm(vec_dip_tip) != 0:
             vec_dip_tip /= np.linalg.norm(vec_dip_tip)
        tip_pos = dip_pos + vec_dip_tip * self.SEGMENT_LENGTHS["DIP_TIP"]
        
        return np.array([mcp_pos, pip_pos, dip_pos, tip_pos])

    def _update_frame(self, frame_idx: int) -> List[plt.Line2D]:
        """The core animation function, called for each frame."""
        row = self.df.iloc[frame_idx]
        mcp_positions = self._reconstruct_mcp_positions(row)

        for finger_name in self.FINGER_NAMES:
            name_lower = finger_name.lower()
            pip_angle = row.get(f'angle_{name_lower}_pip', 180)
            dip_angle = row.get(f'angle_{name_lower}_dip', 180)
            mcp_pos = mcp_positions[finger_name]
            
            if finger_name == "THUMB":
                thumb_palm_dist = row.get('thumb_palm_dist', 0)
                adduction_angle = row.get('thumb_adduction_angle', 45)
                base_dir = np.array([np.sin(np.radians(adduction_angle)), -np.cos(np.radians(adduction_angle)), thumb_palm_dist * 2.0])
                # --- FIX: Initialize as a float array ---
                up_vec = np.array([-1.0, -1.0, 0.0])
            else:
                base_dir = np.array([0.0, 0.1, 1.0]) 
                # --- FIX: Initialize as a float array ---
                up_vec = np.array([-1.0, 0.0, 0.0])

            if np.linalg.norm(base_dir) != 0:
                base_dir /= np.linalg.norm(base_dir)
            if np.linalg.norm(up_vec) != 0:
                up_vec /= np.linalg.norm(up_vec)

            coords = self._reconstruct_finger(mcp_pos, pip_angle, dip_angle, base_dir, up_vec)
            
            self.lines[finger_name].set_data(coords[:, 0], coords[:, 1])
            self.lines[finger_name].set_3d_properties(coords[:, 2])

        self.ax.set_title(f"3D Hand Gesture Reconstruction (Frame {frame_idx})")
        return list(self.lines.values())

    def run_animation(self):
        """Starts and displays the 3D animation."""
        ani = FuncAnimation(
            self.fig, self._update_frame, frames=len(self.df),
            interval=50, blit=False
        )
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        visualizer = HandGestureVisualizer(csv_path="./data/session3/hand_features.csv")
        visualizer.run_animation()
    except FileNotFoundError:
        print(f"\nDemonstration failed. Could not find the CSV file at ./data/session3/hand_features.csv")
    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")