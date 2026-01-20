@echo off
echo [INFO] Starting HyperQuant AI Platform...

echo [1/2] Launching Backend (FastAPI)...
start "HyperQuant Backend" cmd /k "cd backend && python -m uvicorn main:app --reload"

echo [2/2] Launching Frontend (Next.js)...
start "HyperQuant Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ==================================================
echo   🚀 Platform is initializing!
echo   
echo   - Frontend: http://localhost:3000
echo   - Backend:  http://localhost:8000/docs
echo ==================================================
pause
