# Base image: Debian "slim" Linux with Python 3.12 already installed
FROM python:3.12-slim

# All following commands run inside /app (created if it doesn't exist)
WORKDIR /app

# Install the Python dependencies first: this layer is cached and only rebuilt
# when requirements.txt changes, not every time the code changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code and the trained model into /app
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Documentation: the service listens on port 9696 inside the container
EXPOSE 9696

# Start the production web server when a container starts
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
