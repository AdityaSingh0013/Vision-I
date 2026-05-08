import cv2
import numpy as np


def preprocess_for_ocr(image, target_width=1280):
    """
    Preprocess image for OCR accuracy.
    """

    height, width = image.shape[:2]

    # Resize
    scale = target_width / width
    resized = cv2.resize(
        image,
        (target_width, int(height * scale))
    )

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)

    # Threshold
    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return processed