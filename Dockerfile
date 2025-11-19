FROM python:3.12-slim

# Set working directory
WORKDIR /app


# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Flask port
EXPOSE 5001

# Run the app
CMD ["python", "app.py"]
