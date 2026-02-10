# --- Stage 1: Build the Frontend ---
    FROM node:20-slim as frontend-builder

    # Install pnpm (required for your project)
    RUN npm install -g pnpm
    
    # Set working directory for frontend
    WORKDIR /app/frontend
    
    # Copy frontend dependency files
    COPY frontend/package.json frontend/pnpm-lock.yaml ./
    
    # Install dependencies
    RUN pnpm install --frozen-lockfile
    
    # Copy the rest of the frontend source code
    COPY frontend/ ./
    
    # Build the static files (creates /app/frontend/dist)
    RUN pnpm run build
    
    # --- Stage 2: Build the Backend ---
    FROM python:3.12-slim as backend
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Install uv
    RUN pip install uv
    
    # Set working directory
    WORKDIR /app
    
    # Copy requirements and install Python dependencies
    COPY src/requirements.txt ./src/
    RUN uv pip install --system -r src/requirements.txt
    
    # Copy source code
    COPY src/ ./src/
    
    # CRITICAL STEP: Copy the built frontend from Stage 1
    COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
    
    # Expose port
    EXPOSE 8000
    
    # Set environment variables
    ENV PYTHONPATH="/app"
    
    # Command to run the application
    CMD ["uv", "run", "fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]