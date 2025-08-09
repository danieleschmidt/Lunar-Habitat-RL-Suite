# Lunar Habitat RL Production Dockerfile
FROM python:3.12-slim

LABEL maintainer="Terragon Labs <contact@terragon-labs.ai>"
LABEL description="Lunar Habitat Reinforcement Learning Suite - Production Ready"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash habitat
RUN chown -R habitat:habitat /app
USER habitat

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data /app/results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from lunar_habitat_rl import LunarHabitatEnv; env = LunarHabitatEnv(physics_enabled=False); env.close()" || exit 1

# Expose ports
EXPOSE 8080 8443

# Default command
CMD ["python", "-m", "lunar_habitat_rl.server"]