from fastapi import APIRouter

from src.api.models.ocr_models import (
    OCRRequest,
    OCRResponse
)

from src.services.ocr_service import OCRService


router = APIRouter()

ocr_service = OCRService()


@router.post(
    "/ocr",
    response_model=OCRResponse,
    tags=["OCR"]
)
async def ocr_endpoint(request: OCRRequest):
    """
    Extract readable text from image
    for visually impaired accessibility narration.
    """

    result = await ocr_service.extract_text(
        frame_b64=request.frame_b64,
        frame_width=request.frame_width,
        read_mode=request.read_mode
    )

    return result