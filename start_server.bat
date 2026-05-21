@echo off
REM Démarrage du serveur FastAPI RGPH 2014

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         RGPH 2014 - Serveur Frontend + Backend                 ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou n'est pas dans le PATH
    pause
    exit /b 1
)

REM Installer les dépendances si nécessaire
echo Vérification des dépendances...
pip install -q fastapi uvicorn pydantic pandas numpy joblib python-multipart >nul 2>&1

REM Démarrer le serveur
echo.
echo 🌐 Dashboard   : http://localhost:8000/
echo 📚 API Docs    : http://localhost:8000/docs
echo 🔄 API ReDoc   : http://localhost:8000/redoc
echo.
echo ✅ Serveur démarrant...
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
