from pydantic_settings import BaseSettings

from eyegoogler.model.googly_eye_generator import GooglyEyeConfig


class EyeGooglerConfig(BaseSettings):
    assets_path: str = "assets/"
    detection_model_file: str = "blaze_face_short_range.tflite"
    googly_eye_config: GooglyEyeConfig = GooglyEyeConfig()
