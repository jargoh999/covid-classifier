# # Use an official Python runtime as a parent image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Download the model during image build
# RUN python -c "import app; app.download_model()"

# # Make port 8000 available to the world outside this container
# EXPOSE 8000

# # Define environment variable to run in production
# ENV FLASK_ENV=production

# # Run app.py when the container launches
# CMD ["python", "app.py"]
# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download model during build
RUN python -c "import app; app.download_model()"

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    TENSORFLOW_FORCE_GPU_ALLOW_GROWTH=true

# Use gunicorn with limited workers and threads
CMD ["gunicorn", \
    "--workers=2", \
    "--threads=4", \
    "--worker-tmp-dir=/dev/shm", \
    "-b", "0.0.0.0:8000", \
    "app:app"]