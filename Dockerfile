# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy pyproject.toml first for better caching
COPY --chown=app:app pyproject.toml ./

# Install Python dependencies from pyproject.toml
RUN pip install --user --no-cache-dir .

# Copy application code
COPY --chown=app:app . .

# Set environment variables
ENV FLASK_APP=app.main:app
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app.main:app"]
