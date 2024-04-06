import cv2
import numpy as np

from eyegoogler.model.eye_detector import EyeDetector


def test_detection(data_dir):
    detector = EyeDetector(data_dir / "blaze_face_short_range.tflite")
    img = cv2.imread(str(data_dir / "image.jpg"))
    actual = detector.detect(img)

    # Expect 2 faces detected
    assert len(actual) == 2


def test__relative_coords_to_px():
    actual = EyeDetector._relative_coords_to_px(
        img_array=np.ones((200, 500, 3)), x=0.33, y=0.5
    )
    assert actual == (165, 100)
