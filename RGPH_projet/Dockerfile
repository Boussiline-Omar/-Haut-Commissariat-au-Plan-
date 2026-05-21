FROM python:3.10-slim

WORKDIR /app

# Copier les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY main.py .
COPY rgph_dashboard.html .
COPY outputs/ outputs/

# Exposer le port
EXPOSE 8000

# Démarrer le serveur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
