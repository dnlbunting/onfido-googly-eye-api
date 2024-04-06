from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from eyegoogler.api.config import EyeGooglerConfig
from eyegoogler.model.eye_detector import EyeDetector
from eyegoogler.model.googly_eye_generator import GooglyEyeGenerator

MODEL_CONTEXT = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = EyeGooglerConfig()
    MODEL_CONTEXT["eye_detector"] = EyeDetector(
        model=Path(config.assets_path) / config.detection_model_file
    )
    MODEL_CONTEXT["eye_generator"] = GooglyEyeGenerator(config.googly_eye_config)
    yield
    MODEL_CONTEXT.clear()
