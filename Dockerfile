# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 trader
USER trader

# Set environment variables for configuration
ENV BINANCE_API_KEY="" \
    BINANCE_API_SECRET="" \
    NEWS_API_KEY="" \
    RISK_TOLERANCE=0.5 \
    PORTFOLIO_SIZE=1000 \
    MEMORY_DECAY_RATE=0.9

# Run the trading agent
CMD ["python", "-m", "src.main"] 