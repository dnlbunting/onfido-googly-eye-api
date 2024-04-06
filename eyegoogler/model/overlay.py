from typing import Tuple

import numpy as np


def overlay_images(
    base_image: np.ndarray, overlay_image: np.ndarray, centre: Tuple[float, float]
) -> np.ndarray:
    """Overlays `overlay_image` onto `base_image` so the centre of `overlay_image` is at pixel indices
    `centre` on `base_image`, respecting the transparency of `overlay_image`

    :param base_image: The background image as a BGR array
    :param overlay_image: The image to overly onto background image a BGRA array
    :param centre: pixel coordinates on background image to center the overlay image
    :return: BGR image array
    """
    mh = int(overlay_image.shape[0] / 2)
    overlay_alpha = overlay_image[:, :, -1]
    slice_y = slice((centre[0] - mh), (centre[0] + mh))
    slice_x = slice((centre[1] - mh), (centre[1] + mh))

    for c in range(3):
        channel = (1 - overlay_alpha) * base_image[slice_x, slice_y, c]
        channel += overlay_alpha * overlay_image[:, :, c]
        base_image[slice_x, slice_y, c] = channel
    return base_image
