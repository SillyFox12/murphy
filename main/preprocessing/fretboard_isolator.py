# -*- coding: utf-8 -*-
"""
Murphy AI - Fretboard Detector using YOLOv11

This script defines a complete pipeline for training, validating, and using a
YOLOv11 model to detect guitar fretboards.

Author: Professor (Assisted by Silly Fox Studios)
Version: 1.1 (Path corrected)
"""

import os
import json
from typing import Optional, List, Tuple

# Ensure you have the necessary packages installed:
# pip install ultralytics roboflow opencv-python
from ultralytics import YOLO
import roboflow
import supervision as sv
import cv2
import requests
import numpy as np

class FretboardDetectorYOLO:
    """
    A class to manage the training, evaluation, and inference of a YOLOv11
    model for guitar fretboard detection.

    This class handles:
    1. Downloading the specified dataset from Roboflow.
    2. Training the model.
    3. Evaluating the final trained model on the test set.
    4. Performing inference on new images.
    """

    def __init__(self,
                 roboflow_api_key: str,
                 project_id: str = "fretboard-lfgvx",
                 workspace: str = "silly-fox-studios",
                 model_variant: str = 'yolo11n.pt',
                 project_name: str = "runs/detect"): # Standard ultralytics project dir
        """
        Initializes the detector, sets up the environment, and downloads the dataset.

        Args:
            roboflow_api_key (str): Your private Roboflow API key.
            project_id (str): The ID of the Roboflow project.
            workspace (str): The Roboflow workspace ID.
            model_variant (str): The YOLO model variant to use (e.g., 'yolo11n.pt').
            project_name (str): The parent directory for training runs.
        """
        if not roboflow_api_key:
            raise ValueError("Roboflow API key is required.")

        self.api_key = roboflow_api_key
        self.project_id = project_id
        self.workspace = workspace
        self.model_variant = model_variant
        self.project_name = project_name

        self.dataset_path: Optional[str] = None
        self.data_yaml_path: Optional[str] = None
        self.trained_model_path: Optional[str] = None
        self.rf = roboflow.Roboflow(api_key=self.api_key)
        self.project = self.rf.workspace(self.workspace).project(self.project_id)

        self._setup_environment()

    def _setup_environment(self):
        """
        Logs into Roboflow and downloads the specified dataset.
        """
        print("--- [Step 1] Setting up Environment & Downloading Dataset ---")
        try:
            rf = self.rf
            project = self.project
            # Get the latest version of the dataset
            versions = project.versions()
            if not versions:
                raise ValueError("No versions found for this Roboflow project.")
            latest_version_number = versions[0].version
            
            print(f"✅ Found Roboflow dataset: {self.project_id}, latest version: {latest_version_number}")

            # Download the dataset in YOLOv8 format (compatible with v11)
            dataset = project.version(latest_version_number).download("yolov8")
            
            self.dataset_path = dataset.location
            self.data_yaml_path = os.path.join(self.dataset_path, "data.yaml")

            if not os.path.isfile(self.data_yaml_path):
                raise FileNotFoundError(f"data.yaml not found in downloaded dataset at {self.data_yaml_path}")
            
            print(f"✅ Dataset downloaded successfully to: {self.dataset_path}")
            print(f"   YAML file located at: {self.data_yaml_path}")

        except Exception as e:
            print(f"❌ Failed to download dataset. Please check your API key, workspace, and project ID.")
            print(f"Error: {e}")
            raise

    def train_model(self, epochs: int = 50, imgsz: int = 640, run_name: str = 'fretboard_train'):
        """
        Trains the YOLOv11 model.

        Args:
            epochs (int): The number of training epochs.
            imgsz (int): The image size for training.
            run_name (str): A specific name for this training run's output folder.
        """
        print(f"\n--- [Step 2] Starting Model Training ---")
        try:
            model = YOLO(self.model_variant)

            print("Starting standard training and validation...")
            model.train(
                data=self.data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                project=self.project_name,
                name=run_name,
                exist_ok=True
            )

            # --- PATH CORRECTION ---
            # The path is constructed based on the project and name arguments from the train call.
            results_dir = os.path.join(self.project_name, run_name)
            weights_dir = os.path.join(results_dir, 'weights')
            best_model_path = os.path.join(weights_dir, 'best.pt')

            if os.path.exists(best_model_path):
                self.trained_model_path = best_model_path
                print(f"✅ Training complete. Best model saved at: {self.trained_model_path}")
            else:
                print(f"❌ Training finished, but the 'best.pt' model could not be found in {weights_dir}.")
                print("   Please check the console output for the exact save location.")

        except Exception as e:
            print(f"❌ An error occurred during training: {e}")
            raise

    def test_model(self):
        """
        Evaluates the final trained model on the test set to get performance metrics.
        """
        if not self.trained_model_path:
            print("❌ Model has not been trained yet. Please run `train_model()` first.")
            return

        print("\n--- [Step 3] Evaluating Model on Test Set ---")
        try:
            model = YOLO(self.trained_model_path)
            metrics = model.val(split='test')
            print("✅ Evaluation complete.")
            return metrics

        except Exception as e:
            print(f"❌ An error occurred during evaluation: {e}")
            raise

    def detect_fretboard(self, image_source: str, confidence_threshold: int = 40) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Runs inference using Roboflow-hosted model and returns annotated image + labels.

        Args:
            image_source (str): Path or URL to the image.
            confidence_threshold (int): Minimum confidence % for detections.

        Returns:
            annotated_image (np.ndarray): Image with masks and labels drawn.
            labels (List[str]): List of detected class names with confidence.
        """
        print(f"\n--- [Step 4] Performing Inference on: {image_source} ---")
        
        # 1. Load image (supervision handles URL vs. local)
        try:
            # Attempt to read the image directly
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Could not read image from {image_source}. Check the path or URL.")
        except:
            # If the image is a URL, download it
            response = requests.get(image_source)
            with open("temp_image.jpg", "wb") as f:
                f.write(response.content)
            image = cv2.imread("temp_image.jpg")
            if image is None:
                raise ValueError(f"Could not read image from {image_source} after download. Check the URL or file format.")
        
        try:
            # 2. Runs inference on the image
            print(f"Running inference with confidence threshold: {confidence_threshold}%") 
            model = self.project.version(2).model  # Use the latest version of the model
            results = model.predict(image, confidence=confidence_threshold).json()

            # 3. Check for errors in the response
            if "error" in results:
                raise ValueError(f"Roboflow API error: {results['error']}")

            # 4. Convert to Detections object
            detections = sv.Detections.from_inference(results)

            # 2. Find the first detection labeled 'neck'
            neck_detection_index = -1
            for i, (_, _, _, _, _, data) in enumerate(detections):
                if data.get("class_name", "").lower() == "neck":
                    neck_detection_index = i
                    break
            
            if neck_detection_index == -1:
                print("⚠️ No 'neck' detected in the image.")
                return image, None

            # 3. Extract the segmentation mask for the detected neck
            # The .mask attribute contains all masks, so we select the one at our index
            neck_mask = detections.mask[neck_detection_index]

            # 4. Find the largest contour from the mask
            # The mask is a boolean array, convert it to uint8 for findContours
            contours, _ = cv2.findContours(neck_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("⚠️ Could not find contours in the segmentation mask.")
                return image, None
            
            main_contour = max(contours, key=cv2.contourArea)

            # 5. Calculate the minimum area rotated rectangle
            rotated_rect = cv2.minAreaRect(main_contour)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.intp(box_points) # np.intp is an alias for np.int64 or np.int32

            # 6. Create the annotated image
            annotated_image = image.copy()
            # Draw the precise contour
            cv2.drawContours(annotated_image, [main_contour], -1, (255, 0, 0), 2) # Blue contour
            # Draw the new, tight rotated box
            cv2.drawContours(annotated_image, [box_points], 0, (0, 255, 0), 2) # Green rotated box

            # 7. Warp and crop the new ROI
            center, size, angle = rotated_rect
            width, height = map(int, size)

            # Ensure width is the longer side for consistent orientation
            if width < height:
                angle += 90
                width, height = height, width

            # Get the rotation matrix and warp the image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            warped = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # Crop the straightened fretboard from the warped image
            cropped_roi = cv2.getRectSubPix(warped, (width, height), center)

            print("✅ Successfully generated rotated ROI.")
            return annotated_image, cropped_roi

        except Exception as e:
            print(f"❌ An error occurred during detection and cropping: {e}")
            return None, None



# ==============================================================================
# DEMONSTRATION BLOCK
# ==============================================================================
if __name__ == '__main__': 
    # --- Configuration ---
    # IMPORTANT: Replace with your actual Roboflow API Key if needed.
    ROBOFLOW_API_KEY = "9lkRaWHTPNRBK3vMVVFL" 

    # --- Main Execution ---
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY":
        print("="*60)
        print("🛑 PLEASE CONFIGURE YOUR ROBOFLOW API KEY 🛑")
        print("="*60)
    else:
        # 1. Specify model path
        #trained_model_path = "data/fretboard/weights/best.pt" 

        # 2. Initialize the detector
        detector = FretboardDetectorYOLO(roboflow_api_key=ROBOFLOW_API_KEY)

        # 4. Perform inference on a sample image
        sample_image_url = "https://cdn-images.dzcdn.net/images/artist/f2546a666d757e11fe9b3c9dc1a253d0/500x500-000000-80-0-0.jpg"
        annotated_img, roi_crop = detector.detect_fretboard(sample_image_url)

        if annotated_img is not None:
            cv2.imshow("Annotated Image with Rotated Box", annotated_img)
            cv2.waitKey(0)

        if roi_crop is not None:
            cv2.imshow("Tightly Cropped Fretboard ROI", roi_crop)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

      


