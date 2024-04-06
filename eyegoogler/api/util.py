import io

import cv2
import numpy as np
from fastapi import Response
from fastapi.responses import PlainTextResponse


def make_image_response(img: np.ndarray) -> Response:
    """Encode an image array as a PNG image and wrap in a Response object

    :param img: Image array
    :return: Response object containing img as a png file
    """
    is_success, buffer = cv2.imencode(".png", img)
    if not is_success:
        raise Exception("Error creating output image")
    io_buf = io.BytesIO(buffer)
    return PlainTextResponse(io_buf.read(), media_type="image/png")
