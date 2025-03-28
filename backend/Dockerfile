FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Additional dependencies that might be needed
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
