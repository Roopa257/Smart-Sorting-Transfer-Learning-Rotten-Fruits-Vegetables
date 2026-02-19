# Use official Python base image (choose compatible Python version, e.g. 3.9)
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app1.py", "--server.port=8501", "--server.address=0.0.0.0"]
