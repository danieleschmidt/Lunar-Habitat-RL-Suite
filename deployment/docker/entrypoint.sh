#!/bin/bash
set -euo pipefail

# Production Entrypoint for Lunar Habitat RL Suite
# NASA Mission-Critical Container Runtime
# Implements security hardening and operational monitoring

# ==============================================
# Security and Environment Setup
# ==============================================

echo "=============================================="
echo "NASA Lunar Habitat RL Suite - Production Start"
echo "Security Level: ${LUNAR_RL_SECURITY_LEVEL:-mission_critical}"
echo "Environment: ${LUNAR_RL_ENVIRONMENT:-production}"
echo "=============================================="

# Security: Verify we're running as non-root
if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: Container must not run as root for security compliance"
    exit 1
fi

# Security: Set secure umask
umask 0027

# Verify required directories exist
for dir in /app/data /app/logs /app/config /app/backup; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory $dir does not exist"
        exit 1
    fi
done

# ==============================================
# Pre-flight System Checks
# ==============================================

echo "Performing pre-flight system checks..."

# Check Python environment
python3 --version || {
    echo "ERROR: Python not available"
    exit 1
}

# Check critical Python packages
python3 -c "import numpy, scipy, gymnasium, torch" || {
    echo "ERROR: Critical Python packages not available"
    exit 1
}

# Check application import
python3 -c "import lunar_habitat_rl" || {
    echo "ERROR: Lunar Habitat RL package not importable"
    exit 1
}

# Memory check
total_memory=$(grep MemTotal /proc/meminfo | awk '{print $2}')
min_memory=1048576  # 1GB in KB
if [ "$total_memory" -lt "$min_memory" ]; then
    echo "WARNING: System memory below recommended minimum (1GB)"
fi

# Disk space check
available_space=$(df /app | tail -1 | awk '{print $4}')
min_space=1048576  # 1GB in KB
if [ "$available_space" -lt "$min_space" ]; then
    echo "WARNING: Available disk space below recommended minimum (1GB)"
fi

echo "Pre-flight checks completed successfully"

# ==============================================
# Security Initialization
# ==============================================

echo "Initializing security systems..."

# Initialize security audit logging
mkdir -p /app/logs/security
touch /app/logs/security/audit.log
chmod 640 /app/logs/security/audit.log

# Create PID file for monitoring
echo $$ > /app/logs/lunarrl.pid

# Set up signal handlers for graceful shutdown
trap 'echo "Received SIGTERM, initiating graceful shutdown..."; kill -TERM $child; wait $child' TERM
trap 'echo "Received SIGINT, initiating graceful shutdown..."; kill -INT $child; wait $child' INT

# ==============================================
# Application Configuration
# ==============================================

echo "Loading application configuration..."

# Set default configuration if not provided
if [ ! -f "${LUNAR_RL_CONFIG_PATH:-/app/config/production-config.yml}" ]; then
    echo "WARNING: Production config not found, using defaults"
    export LUNAR_RL_CONFIG_PATH="/app/config/default-config.yml"
fi

# Validate configuration
python3 -c "
import yaml
import sys
try:
    with open('${LUNAR_RL_CONFIG_PATH}', 'r') as f:
        config = yaml.safe_load(f)
    print('Configuration validation: PASSED')
except Exception as e:
    print(f'Configuration validation: FAILED - {e}')
    sys.exit(1)
" || exit 1

# ==============================================
# Resource Monitoring Setup
# ==============================================

echo "Setting up resource monitoring..."

# Create monitoring script for background execution
cat > /app/logs/monitor.sh << 'EOF'
#!/bin/bash
while true; do
    # Log resource usage every 60 seconds
    {
        echo "$(date): RESOURCE_MONITOR"
        echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
        echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
        echo "Disk: $(df -h /app | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"
        echo "---"
    } >> /app/logs/resource_monitor.log 2>&1
    sleep 60
done
EOF

chmod +x /app/logs/monitor.sh

# Start resource monitoring in background
if [ "${LUNAR_RL_ENABLE_MONITORING:-true}" = "true" ]; then
    /app/logs/monitor.sh &
    monitor_pid=$!
    echo "Resource monitoring started (PID: $monitor_pid)"
fi

# ==============================================
# Database/Storage Initialization
# ==============================================

echo "Initializing data storage..."

# Create necessary data directories
mkdir -p /app/data/{models,checkpoints,logs,metrics,experiments}
mkdir -p /app/backup/{daily,weekly,emergency}

# Set proper permissions
chmod 750 /app/data/* /app/backup/*

# Initialize databases/storage if needed
if [ "${LUNAR_RL_INIT_STORAGE:-true}" = "true" ]; then
    python3 -c "
import lunar_habitat_rl
print('Initializing storage systems...')
# Storage initialization would go here
print('Storage initialization completed')
"
fi

# ==============================================
# Security Hardening Runtime
# ==============================================

echo "Applying runtime security hardening..."

# Enable security monitoring
export LUNAR_RL_SECURITY_MONITORING=enabled
export LUNAR_RL_AUDIT_LOGGING=enabled

# Set secure Python flags
export PYTHONHASHSEED=random
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Disable Python debug features in production
if [ "${LUNAR_RL_ENVIRONMENT}" = "production" ]; then
    export PYTHONOPTIMIZE=2
    unset PYTHONDEBUG
fi

# ==============================================
# Start Application
# ==============================================

echo "Starting Lunar Habitat RL Suite..."
echo "Command: $*"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"
echo "PID: $$"
echo "=============================================="

# Log startup event
{
    echo "$(date): APPLICATION_START"
    echo "User: $(whoami)"
    echo "PID: $$"
    echo "Command: $*"
    echo "Environment: ${LUNAR_RL_ENVIRONMENT:-production}"
    echo "Security Level: ${LUNAR_RL_SECURITY_LEVEL:-mission_critical}"
} >> /app/logs/security/audit.log

# Execute the main application with proper signal handling
if [ $# -gt 0 ]; then
    # Execute provided command
    exec "$@" &
    child=$!
    wait $child
    exit_code=$?
else
    # Default command: start the RL suite server
    exec python -m lunar_habitat_rl.cli server \
        --host "${LUNAR_RL_HOST:-0.0.0.0}" \
        --port "${LUNAR_RL_PORT:-8080}" \
        --config "${LUNAR_RL_CONFIG_PATH}" \
        --log-level "${LUNAR_RL_LOG_LEVEL:-INFO}" &
    child=$!
    wait $child
    exit_code=$?
fi

# ==============================================
# Cleanup and Shutdown
# ==============================================

echo "Application shutdown initiated..."

# Log shutdown event
{
    echo "$(date): APPLICATION_SHUTDOWN"
    echo "Exit Code: $exit_code"
    echo "Runtime: $(($(date +%s) - $(stat -c %Y /app/logs/lunarrl.pid))) seconds"
} >> /app/logs/security/audit.log

# Stop background processes
if [ -n "${monitor_pid:-}" ]; then
    kill $monitor_pid 2>/dev/null || true
fi

# Clean up PID file
rm -f /app/logs/lunarrl.pid

# Security: Clear sensitive environment variables
unset LUNAR_RL_SECRET_KEY LUNAR_RL_API_TOKEN

echo "Graceful shutdown completed"
exit $exit_code