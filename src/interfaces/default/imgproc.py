"""Image resizing for the bundled default detector."""

import cv2
import numpy as np


def resize_aspect_ratio(image, square_size):
    height, width, channels = image.shape
    ratio = square_size / max(height, width)
    target_height = int(round(height * ratio))
    target_width = int(round(width * ratio))
    resized_image = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_height = (-target_height) % 256
    pad_width = (-target_width) % 256
    padded = np.zeros(
        (
            target_height + pad_height,
            target_width + pad_width,
            channels,
        ),
        dtype=np.uint8,
    )
    padded[:target_height, :target_width] = resized_image
    return padded, ratio, pad_width, pad_height
