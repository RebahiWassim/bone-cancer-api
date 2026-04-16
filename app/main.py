import logging
from fastapi import FastAPI
from app.api.routes import predict
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Initialiser le modèle au démarrage
@app.on_event("startup")
async def startup_event():
    predict.init_model()
    logger.info(f"API started on {settings.DEVICE}")

# Inclure les routes
app.include_router(predict.router, tags=["prediction"])

@app.get("/")
async def root():
    return {
        "status": "ok",
        "device": str(settings.DEVICE),
        "classes": settings.CLASS_NAMES
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}