"""
Local test for Vision I services layer.
"""

import base64
import asyncio
import cv2

from src.services.decision_service import (
    run_decision,
)
from src.services.describe_service import (
    run_describe,
)
from src.services.detection_service import (
    run_detection,
)
from src.services.model_loader import (
    load_model,
)
from src.services.ocr_service import (
    OCRService,
)

# =========================================================
# CONFIG
# =========================================================

TEST_IMAGE = "cache/test.jpg"
OCR_TEST_IMAGE = "cache/ocr_test.jpg"

# =========================================================
# STARTUP
# =========================================================

print("[INFO] Loading model...")

load_model()

ocr_service = OCRService()

print("[INFO] Model loaded")

# =========================================================
# LOAD IMAGE
# =========================================================

frame = cv2.imread(TEST_IMAGE)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {TEST_IMAGE}"
    )

print("[INFO] Image loaded")

# =========================================================
# DETECTION TEST
# =========================================================

print("\n===== DETECTION TEST =====")

detection_result = run_detection(
    frame=frame,
    frame_width=640,
)

print(detection_result)

# =========================================================
# DECISION TEST
# =========================================================

print("\n===== DECISION TEST =====")

decision_result = run_decision(
    detections=detection_result["detections"],
)

print(decision_result)

# =========================================================
# OCR TEST
# =========================================================

print("\n===== OCR TEST =====")

with open(OCR_TEST_IMAGE, "rb") as image_file:
    frame_b64 = base64.b64encode(
        image_file.read()
    ).decode("utf-8")

ocr_result = asyncio.run(
    ocr_service.extract_text(
        frame_b64=frame_b64,
        frame_width=640,
        read_mode="standard"
    )
)

print(ocr_result)

# =========================================================
# DESCRIPTION TEST
# =========================================================

print("\n===== DESCRIPTION TEST =====")

description = run_describe(
    frame=frame,
    mode="awareness",
)

print(description)

print("\n[INFO] ALL TESTS COMPLETED")