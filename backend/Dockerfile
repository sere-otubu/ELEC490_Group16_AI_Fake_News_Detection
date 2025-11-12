# Step 1: Start from an official Python base image
# This gives us a clean Linux environment with Python 3.11
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
# This is where our app's files will live
WORKDIR /app

# Step 3: Copy the requirements file in first
# We copy this separately so Docker can cache our installed libraries,
# making future builds much faster.
COPY requirements.txt .

# Step 4: Install the Python dependencies
# This runs 'pip install -r requirements.txt' just like you did locally
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your backend application code
# This copies 'api.py' and any other files into the /app directory
COPY . .

# Step 6: Expose the port your app runs on
# This tells Docker that our FastAPI app (uvicorn) will be running on port 8000
EXPOSE 8000

# Step 7: Define the command to run your application
# This is the command that starts your server when the container launches
# It's the same as 'python api.py' but uses the uvicorn command directly
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]