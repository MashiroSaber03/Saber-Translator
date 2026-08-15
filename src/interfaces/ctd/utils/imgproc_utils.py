"""Image resizing used by the current CTD inference path."""

import cv2


def letterbox(image, new_shape=(640, 640), color=(0, 0, 0)):
    height, width = image.shape[:2]
    target_height, target_width = (
        new_shape if isinstance(new_shape, tuple) else (new_shape, new_shape)
    )
    ratio = min(target_height / height, target_width / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    image = cv2.copyMakeBorder(
        image,
        0,
        pad_height,
        0,
        pad_width,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return image, (ratio, ratio), (pad_width, pad_height)
