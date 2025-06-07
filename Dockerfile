FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Create cache directory for FastEmbed models
RUN mkdir -p /root/.cache/fastembed && chmod 755 /root/.cache/fastembed

# Copy model download script and run it
COPY download_models.py .
RUN python download_models.py

# Copy application code
COPY . .

# Copy and make startup script executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Command to run both FastAPI and Streamlit
CMD ["/app/start.sh"] 