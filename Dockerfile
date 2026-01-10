# Dockerfile for ELEC 498 Capstone - Medical Misinformation Detection

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy uv files (if you have them, otherwise skip these lines)
# COPY pyproject.toml uv.lock ./

# Copy requirements.txt and install dependencies
COPY src/requirements.txt ./src/

# Install Python dependencies using uv
RUN uv pip install --system -r src/requirements.txt

# Copy source code
COPY src/ ./src/

# Copy data folder for document processing
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH="/app"

# Command to run the application
CMD ["uv", "run", "fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]