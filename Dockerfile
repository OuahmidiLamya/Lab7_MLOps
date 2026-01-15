FROM python:3.10-slim

# 1) Dossier de travail
WORKDIR /app

# 2) Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copier le projet
COPY . .

# 4) Exposer le port de l’API
EXPOSE 8000

# 5) Lancer l’API FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
