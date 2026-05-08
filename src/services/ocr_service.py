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
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

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
            image = self._decode_image(frame_b64)

            if image is None:
                return {
                    "success": False,
                    "error": "Invalid image data"
                }

            processed = preprocess_for_ocr(
                image,
                target_width=frame_width * 2
            )

            loop = asyncio.get_running_loop()

            results = await loop.run_in_executor(
                None,
                lambda: self.reader.readtext(processed)
            )

            if not results:
                return {
                    "success": False,
                    "error": "No readable text detected"
                }

            raw_text, confidence = self._parse_results(results)

            cleaned_text = self._clean_text(raw_text)

            reading_time = self._estimate_reading_time(cleaned_text)

            return {
                "success": True,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "language": "en",
                "confidence": round(confidence, 2),
                "speak": True,
                "reading_time_seconds": reading_time
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # --------------------------------------------------

    def _decode_image(self, frame_b64):
        """
        Decode base64 image.
        """

        try:
            image_bytes = base64.b64decode(frame_b64)

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

    # --------------------------------------------------

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

    # --------------------------------------------------

    def _clean_text(self, text):
        """
        Accessibility-focused OCR cleanup.
        """

        text = re.sub(r'\s+', ' ', text)

        text = re.sub(
            r'[^a-zA-Z0-9\s.,!?;:\'-]',
            '',
            text
        )

        text = text.strip()

        return text

    # --------------------------------------------------

    def _estimate_reading_time(self, text):
        """
        Estimate narration duration.
        """

        words = len(text.split())

        wpm = 130

        minutes = words / wpm

        return max(1, math.ceil(minutes * 60))