#!/bin/bash

# activate your virtualenv if you have one
source ./venv/bin/activate 

# ========================================
# Run backend (FastAPI)
# ========================================
echo "Starting FastAPI backend..."
# cd app/backend
# Optional: activate virtualenv if you have one
# source venv/bin/activate

# Run backend in background so frontend can start too
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# ========================================
# Run frontend (React)
# ========================================
echo "Starting React frontend..."
cd app/frontend
npm install   # install dependencies if not done yet
npm start     # default runs on port 3000

# ========================================
# Cleanup on exit
# ========================================
echo "Stopping backend..."
kill $BACKEND_PID
