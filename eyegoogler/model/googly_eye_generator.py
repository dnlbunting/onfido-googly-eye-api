from typing import Tuple

from pydantic import BaseModel
import numpy as np
import cv2


class GooglyEyeConfig(BaseModel):
    RADUIS1: int = 450
    RADUIS2: int = 250
    RADUIS3: int = 40
    CENTER: Tuple[int, int] = (500, 500)
    OFFSET1: int = 200
    OFFSET2X: int = 115
    OFFSET2Y: int = -60

    size_scale: float = 0.75


class GooglyEyeGenerator:
    """The GooglyEyeGenerator is responsible for generating the random images of the googly eyes"""

    def __init__(self, config: GooglyEyeConfig):
        self.config = config
        self.eye_template = self.generate_eye_template()

    def generate_eye_template(self) -> np.ndarray:
        """Generates a template googly eye based on the config parameters
        :return: Image of a googly eye as a BGRA array
        """
        # Set up a transparent background canvas
        eye = 255 * np.ones((1000, 1000, 4), dtype=np.int8)
        eye[:, :, -1] = 0

        # Draws a filled white circle with a black outline for the iris
        eye = cv2.circle(
            eye, self.config.CENTER, self.config.RADUIS1, (255, 255, 255, 1), -1
        )
        eye = cv2.circle(eye, self.config.CENTER, self.config.RADUIS1, (0, 0, 0, 1), 40)

        # Draws a filled black circle for the pupil
        eye = cv2.circle(
            eye,
            (self.config.CENTER[0] + self.config.OFFSET1, self.config.CENTER[1]),
            self.config.RADUIS2,
            (0, 0, 0, 1),
            -1,
        )

        # Draws a filled white circle for the pupil highlight
        eye = cv2.circle(
            eye,
            (
                self.config.CENTER[0] + self.config.OFFSET1 + self.config.OFFSET2X,
                self.config.CENTER[1] + self.config.OFFSET2Y,
            ),
            self.config.RADUIS3,
            (255, 255, 255, 1),
            -1,
        )
        return eye

    @staticmethod
    def rotate(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotates an input image array by the specified angle

        :param image: image array
        :param angle: angle to rotate in degrees
        :return: image array
        """
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def generate(self, intra_eye_distance: float) -> np.ndarray:
        """Generates an image of a googly eye, appropriately scaled for the specified
        distance between the eyes, based on the config.size_scale. With some randomisation of
        the size and orientation of the eye for comedic effect.

        :param intra_eye_distance: Measured distance between the eyes in the target image
        :return: BGRA image array
        """
        angle = np.random.randint(0, 360)
        scale = self.config.size_scale * np.random.uniform(0.75, 1.25)
        eye_size = 2 * (int(intra_eye_distance * scale) // 2)
        eye = cv2.resize(self.eye_template, (eye_size, eye_size))
        return self.rotate(eye, angle)
