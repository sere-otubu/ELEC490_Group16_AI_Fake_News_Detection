# Start from a standard Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file in first
COPY requirements.txt .

# Install the Python dependencies
# We use the --extra-index-url from your requirements file
RUN pip install \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r requirements.txt

# Copy the rest of your application code
COPY main.py .

# Expose port 8000 (which uvicorn will run on)
EXPOSE 8000

# The command to run your API when the container starts
# This runs the 'app' object inside the 'main.py' file
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]