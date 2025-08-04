#!/bin/bash

echo "==============================="
echo " 🚀  Lancement du serveur FastAPI"
echo "==============================="

uvicorn main:app --reload --host 0.0.0.0 --port 8081
