import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, Response, status

from eyegoogler.api.lifespan import lifespan, MODEL_CONTEXT
from eyegoogler.api.util import make_image_response
from eyegoogler.model.eye_detector import EyeDetector
from eyegoogler.model.googly_eye_generator import GooglyEyeGenerator
from eyegoogler.model.overlay import overlay_images

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return "OK"


@app.post("/eyegoogler/")
def eyegoogler(file: UploadFile, response: Response):
    """API endpoint that recieves image data via UploadFile and returns the same image with funny
    googly eyes covering all identified eyes in the target image
    """
    detector: EyeDetector = MODEL_CONTEXT["eye_detector"]
    generator: GooglyEyeGenerator = MODEL_CONTEXT["eye_generator"]

    try:
        img = cv2.imdecode(buf=np.frombuffer(file.file.read()), flags=cv2.IMREAD_COLOR)
    except cv2.error:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return "Error - invalid image provided"

    eye_key_points = detector.detect(img)

    for face in eye_key_points:
        intra_eye_distance = np.sqrt(
            ((np.array(face.right) - np.array(face.left)) ** 2).sum()
        )
        left_eye = generator.generate(intra_eye_distance)
        right_eye = generator.generate(intra_eye_distance)
        img = overlay_images(base_image=img, overlay_image=left_eye, centre=face.left)
        img = overlay_images(base_image=img, overlay_image=right_eye, centre=face.right)

    return make_image_response(img)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
