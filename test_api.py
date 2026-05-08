"""
Vision I API test client.
"""

import base64
import json

import requests

# =========================================================
# CONFIG
# =========================================================

API_URL = "https://vision-i-api.up.railway.app/ocr"

IMAGE_PATH = "cache/ocr_test.jpg"

# =========================================================
# IMAGE -> BASE64
# =========================================================

with open(IMAGE_PATH, "rb") as image_file:

    frame_b64 = base64.b64encode(
        image_file.read()
    ).decode("utf-8")

# =========================================================
# REQUEST BODY
# =========================================================

payload = {
    "frame_b64": frame_b64,
    "frame_width": 640,
    "read_mode": "standard"
}
# =========================================================
# SEND REQUEST
# =========================================================

response = requests.post(
    API_URL,
    json=payload,
)

# =========================================================
# OUTPUT
# =========================================================

print("\n===== STATUS =====")
print(response.status_code)

print("\n===== RESPONSE =====")

data = response.json()

print(
    json.dumps(
        data,
        indent=2
    )
)