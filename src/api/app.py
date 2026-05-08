"""
Vision I FastAPI application.
"""

from fastapi import FastAPI

from src.api.routes.analyze import (
    router as analyze_router,
)
from src.api.routes.decision import (
    router as decision_router,
)
from src.api.routes.describe import (
    router as describe_router,
)
from src.api.routes.detect import (
    router as detect_router,
)
from src.services.model_loader import (
    load_model,
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Vision I API",
    description=(
        "AI-Based Intelligent Visual Guidance System "
        "for visually impaired users."
    ),
    version="1.0.0",
)

# =========================================================
# STARTUP
# =========================================================


@app.on_event("startup")
async def startup_event():

    print("[INFO] Loading Vision I services...")

    load_model()

    print("[INFO] Vision I API ready")


# =========================================================
# ROOT
# =========================================================


@app.get("/")
async def root():

    return {
        "message": "Vision I API running"
    }


# =========================================================
# ROUTERS
# =========================================================

app.include_router(detect_router)

app.include_router(decision_router)

app.include_router(describe_router)

app.include_router(analyze_router)