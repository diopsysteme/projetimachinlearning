# 1. Utilise une image Python officielle
FROM python:3.10-slim

# 2. Définir le répertoire de travail
WORKDIR /app

# 3. Copier les fichiers nécessaires
COPY . .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exposer le port de Flask
EXPOSE 5000

# 6. Démarrer l’application
CMD ["python", "app.py"]
