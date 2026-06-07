"""FastAPI application factory and shared singletons."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import get_config
from src.utils.logger import setup_logger

# ---- Module-level singletons shared with routes.py -------------------------
config = get_config()
setup_logger("src", log_level=config.log_level)
logger = logging.getLogger(__name__)

detector = None
loader = None


# ---- Lifespan: initialize heavy resources once at startup ------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the detector model on startup; release on shutdown."""
    global detector, loader

    try:
        from src.detection.detector import ProductDetector
        from src.preprocessing.image_loader import ImageLoader

        detector = ProductDetector(
            model=config.detection.model_name,
            confidence_threshold=config.detection.confidence_threshold,
            iou_threshold=config.detection.iou_threshold,
            device=config.detection.device,
        )
        loader = ImageLoader()
        logger.info("Detector initialised successfully.")
    except Exception as exc:
        logger.error("Failed to initialise detector: %s", exc)
        # detector stays None; /health will report unhealthy

    yield  # application runs here

    detector = None
    loader = None
    logger.info("Detector released.")


# ---- Application factory ----------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=config.app_name,
        version=config.version,
        description=(
            "REST API for automated product detection and classification "
            "in ecommerce images."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root redirect
    @application.get("/", include_in_schema=False)
    async def root():
        return {
            "name": config.app_name,
            "version": config.version,
            "docs": "/docs",
            "api": "/api/v1",
        }

    # Mount versioned routes
    from src.api.routes import router
    application.include_router(router, prefix="/api/v1")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
    )
