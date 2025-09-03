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
    createdAt: datetime  # Utiliser datetime, Pydantic le sérialisera en ISO

    class Config:
        from_attributes = True  # Remplace orm_mode pour Pydantic v2
