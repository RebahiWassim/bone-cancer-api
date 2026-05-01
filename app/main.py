import logging
from fastapi import FastAPI
from app.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Colon Cancer AI",
    description="API de prédiction pour le cancer du côlon",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    from app.models.model_loader import load_model
    logger.info("Démarrage et chargement du modèle...")
    load_model()
    logger.info("Application prête!")

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Colon Cancer AI API"}
