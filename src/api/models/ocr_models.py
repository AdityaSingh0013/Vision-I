from pydantic import BaseModel
from typing import Optional


class OCRRequest(BaseModel):
    frame_b64: str
    frame_width: Optional[int] = 640
    read_mode: Optional[str] = "standard"


class OCRResponse(BaseModel):
    success: bool
    raw_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    language: Optional[str] = "en"
    confidence: Optional[float] = None
    speak: bool = True
    reading_time_seconds: Optional[int] = None
    error: Optional[str] = None