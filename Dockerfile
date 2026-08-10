FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Expose the port the app runs on
EXPOSE 5500

# Set the working directory to where app.py is located
WORKDIR /app/src

# Command to run the application
CMD ["python", "app.py"]
