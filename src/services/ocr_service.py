import re
import math
import base64
import asyncio

import cv2
import easyocr
import numpy as np

from src.utils.image_preprocessing import preprocess_for_ocr


class OCRService:
    def __init__(self):
        """
        Lazy-loaded OCR reader.

        Prevents Railway startup OOM crashes.
        """
        self.reader = None

    async def extract_text(
        self,
        frame_b64: str,
        frame_width: int = 640,
        read_mode: str = "standard"
    ):
        """
        Extract readable text from image.
        """

        try:

            # --------------------------------------------------
            # Lazy load EasyOCR only when needed
            # --------------------------------------------------

            if self.reader is None:
                self.reader = easyocr.Reader(
                    ['en'],
                    gpu=False
                )

            # --------------------------------------------------
            # Decode image
            # --------------------------------------------------

            image = self._decode_image(frame_b64)

            if image is None:
                return {
                    "success": False,
                    "error": "Invalid image data"
                }

            # --------------------------------------------------
            # Preprocess image
            # --------------------------------------------------

            processed = preprocess_for_ocr(
                image,
                target_width=frame_width * 2
            )

            # --------------------------------------------------
            # Async OCR execution
            # --------------------------------------------------

            loop = asyncio.get_running_loop()

            results = await loop.run_in_executor(
                None,
                lambda: self.reader.readtext(processed)
            )

            # --------------------------------------------------
            # Empty OCR result
            # --------------------------------------------------

            if not results:
                return {
                    "success": False,
                    "error": "No readable text detected"
                }

            # --------------------------------------------------
            # Parse OCR output
            # --------------------------------------------------

            raw_text, confidence = self._parse_results(results)

            cleaned_text = self._clean_text(raw_text)

            reading_time = self._estimate_reading_time(
                cleaned_text
            )

            # --------------------------------------------------
            # Success response
            # --------------------------------------------------

            return {
                "success": True,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "language": "en",
                "confidence": float(
                    round(confidence, 2)
                ),
                "speak": True,
                "reading_time_seconds": reading_time
            }

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }

    # ==================================================
    # Decode Base64 Image
    # ==================================================

    def _decode_image(self, frame_b64):
        """
        Decode base64 image.
        """

        try:

            image_bytes = base64.b64decode(
                frame_b64
            )

            np_array = np.frombuffer(
                image_bytes,
                np.uint8
            )

            image = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )

            return image

        except Exception:
            return None

    # ==================================================
    # Parse OCR Results
    # ==================================================

    def _parse_results(self, results):
        """
        Merge OCR detections.
        """

        texts = []
        confidences = []

        for result in results:

            text = result[1]
            confidence = result[2]

            texts.append(text)
            confidences.append(confidence)

        raw_text = " ".join(texts)

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        return raw_text, avg_confidence

    # ==================================================
    # Accessibility Cleanup
    # ==================================================

    def _clean_text(self, text):
        """
        Accessibility-focused OCR cleanup.
        """

        # Remove excessive whitespace
        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        # Remove OCR artifacts
        text = re.sub(
            r'[^a-zA-Z0-9\s.,!?;:\'-]',
            '',
            text
        )

        # Cleanup
        text = text.strip()

        return text

    # ==================================================
    # Reading Time Estimation
    # ==================================================

    def _estimate_reading_time(self, text):
        """
        Estimate narration duration.
        """

        words = len(text.split())

        wpm = 130

        minutes = words / wpm

        return max(
            1,
            math.ceil(minutes * 60)
        )