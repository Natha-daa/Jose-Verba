# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Récupère l'URL de la base depuis Render
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Configure it in Render environment variables.")

# Détermine si le mode debug est activé (optionnel via variable d'environnement)
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
engine = create_engine(DATABASE_URL, echo=DEBUG, future=True)

# Test de connexion initiale
try:
    with engine.connect() as connection:
        connection.close()
except SQLAlchemyError as e:
    raise ConnectionError(f"Failed to connect to the database: {str(e)}")

# Création de la session pour interagir avec la DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour tous les modèles
Base = declarative_base()

# Dépendance FastAPI pour récupérer la session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
