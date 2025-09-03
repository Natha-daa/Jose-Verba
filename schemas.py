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
        from_attributes = True  # Remplace orm_mode pour Pydantic v2
        json_encoders = {
            datetime: lambda v: v.isoformat()  # Convertit datetime en chaîne ISO
        }
