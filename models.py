# app/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from db import Base

class Media(Base):
    __tablename__ = "Media"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    createdAt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    audioLink = Column(Text, nullable=True)
    videoLink = Column(Text, nullable=True)
    coverLink = Column(Text, nullable=True)
    fileSize = Column(String, nullable=True)
    numberSpeaker = Column(Integer, nullable=False, default=1)
