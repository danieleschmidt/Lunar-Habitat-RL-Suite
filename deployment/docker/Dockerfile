
# Multi-stage production Dockerfile for Lunar Habitat RL Suite
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Development stage
FROM base as development
RUN pip install --no-cache-dir pytest black flake8 mypy
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
CMD ["python", "-m", "pytest"]

# Production stage
FROM base as production

# Copy application code
COPY --chown=appuser:appuser . .

# Install package
RUN pip install --no-cache-dir -e . \
    && python -m compileall /app

# Security hardening
RUN chown -R appuser:appuser /app \
    && chmod -R 750 /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import lunar_habitat_rl; env = lunar_habitat_rl.make_lunar_env(); env.reset()"

# Expose port
EXPOSE 8000

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "60", "lunar_habitat_rl.api:app"]
