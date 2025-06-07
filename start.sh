#!/bin/bash
set -e

echo "Starting FastAPI..."
echo "Models are pre-downloaded, starting services..."

echo "Starting uvicorn server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info &
FASTAPI_PID=$!

echo "Waiting for FastAPI to start..."
sleep 5

echo "Starting Streamlit..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo "Both services started. FastAPI PID: $FASTAPI_PID, Streamlit PID: $STREAMLIT_PID"

wait 