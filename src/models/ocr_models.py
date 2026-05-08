from pydantic import BaseModel


class OCRRequest(BaseModel):
    frame_b64: str
    frame_width: int = 640
    read_mode: str = "standard"


class OCRResponse(BaseModel):
    success: bool
    raw_text: str | None = None
    cleaned_text: str | None = None
    language: str | None = "en"
    confidence: float | None = None
    speak: bool = True
    reading_time_seconds: int | None = None
    error: str | None = None