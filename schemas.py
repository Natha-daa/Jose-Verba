# schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MediaBase(BaseModel):
    name: str
    description: Optional[str] = None
    audioLink: Optional[str] = None
    videoLink: Optional[str] = None
    coverLink: Optional[str] = None
    fileSize: Optional[str] = None
    numberSpeaker: int

class MediaSchema(MediaBase):
    id: int
    createdAt: str  # Définir comme chaîne pour la réponse JSON

    class Config:
        orm_mode = True  # Permet l'utilisation avec SQLAlchemy
        # Pydantic convertit automatiquement datetime en chaîne ISO avec orm_mode
