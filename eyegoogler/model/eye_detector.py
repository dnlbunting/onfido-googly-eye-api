import dataclasses
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


@dataclasses.dataclass
class EyeDetection:
    """Holds the coordinates in pixels of the left and right eyes on a face"""

    left: Tuple[float, float]
    right: Tuple[float, float]


class EyeDetector:
    """The EyeDetector class is responsible for identifying faces and eyes in images

    It requires access to a pretrained BlazeFace model artefact downloadable from
    https://developers.google.com/mediapipe/solutions/vision/face_detector#blazeface_short-range

    """

    def __init__(self, model: Path):
        """
        :param model: Path to a pretrained BlazeFace model artefact
        """
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    @staticmethod
    def _relative_coords_to_px(
        img_array: np.ndarray, x: float, y: float
    ) -> Tuple[float, float]:
        """Converts normalised relative coordinates to pixel values based on the input image

        :param img_array: Input image as an array
        :param x: Normalised x coord
        :param y: Normalised y coord
        :return: x, y coordinates as pixel indices
        """
        height, width, _ = img_array.shape
        x_px = min(math.floor(x * width), width - 1)
        y_px = min(math.floor(y * height), height - 1)
        return x_px, y_px

    def detect(self, img_array: np.ndarray) -> list[EyeDetection]:
        """Detect all faces and the key points of both eyes within the face for an input image

        :param img_array: Input image as a BGR array
        :return: Eye locations of each face detected in the image
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        detection_result = self.detector.detect(image)

        faces = []
        for face in detection_result.detections:
            left_eye = self._relative_coords_to_px(
                img_array, face.keypoints[0].x, face.keypoints[0].y
            )
            right_eye = self._relative_coords_to_px(
                img_array, face.keypoints[1].x, face.keypoints[1].y
            )
            faces.append(EyeDetection(left=left_eye, right=right_eye))
        return faces
