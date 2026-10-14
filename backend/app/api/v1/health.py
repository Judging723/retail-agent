from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

@router.get("/health", response_model=HealthResponse)
def health_check()->HealthResponse:
    return HealthResponse(
        status="ok",
        service="retail-agent-backend",
        version="0.1.0",
    )



