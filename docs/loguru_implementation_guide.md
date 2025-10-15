# Loguru Implementation Guide for MAIE Project

## Comprehensive Analysis of Loguru Best Practices and Implementation Patterns

### **Based on Official Documentation and MCP Research**

## 1. Core Implementation Patterns

### **1.1 Environment-Aware Configuration**

```python
# Production Configuration with Environment Variables
import os
import sys
from loguru import logger

def configure_logging():
    """
    Configure loguru with environment-aware settings based on environment variables
    """
    # Remove default handler
    logger.remove()

    # Get environment configuration
    environment = os.getenv("ENVIRONMENT", "development")
    log_level = os.getenv("LOG_LEVEL", "DEBUG")
    log_format = os.getenv("LOGURU_FORMAT", "{time} [{level}] {message}")
    diagnose = os.getenv("LOGURU_DIAGNOSE", "False").lower() == "true"
    backtrace = os.getenv("LOGURU_BACKTRACE", "True").lower() == "true"

    # Console handler with JSON serialization for Docker containers
    logger.add(
        sys.stdout,
        level=log_level,
        format=log_format,
        serialize=True,  # Critical for Docker deployments
        diagnose=diagnose,  # Disable diagnose in production
        backtrace=backtrace,
        level=log_level
    )

    # File handler with rotation and compression
    if environment != "docker":
        logger.add(
            "logs/app.log",
            rotation="500 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG"
        )
        logger.add(
            "logs/errors.log",
            rotation="100 MB",
            retention="90 days",
            diagnose=False,
            backtrace=True,
            level="ERROR"
        )

    return logger

# Initialize configuration
logger = configure_logging()
```

### **1.2 Docker Deployment Patterns**

```python
# Docker-Ready Configuration
import os
import signal
import sys
from loguru import logger

# Signal handling for graceful shutdown
def signal_handler(signum, frame):
    logger.info("Shutting down gracefully...")
    logger.complete()  # Critical for Docker containers
    sys.exit(0)

# Configure Docker-ready logging (only to stdout)
logger.remove()
logger.add(sys.stdout,
    serialize=True,
    diagnose=False,
    backtrace=True,
    level="INFO")

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

## 2. Structured Logging with Correlation IDs

### \*\*2.1 Correlation ID Patterns

```python
import uuid
from loguru import logger

class StructuredLogger:
    def __init__(self):
        self.request_id = str(uuid.uuid4())

    def bind_context(self, **kwargs):
        """Add correlation ID and context data"""
        return logger.bind(request_id=self.request_id, **kwargs)

    def patch_context(self, patch_func):
        """Patch all subsequent logs with context data"""
        return logger.patch(patch_func)

    def info(self, message, **extra):
        """Structured info log with extra data"""
        return self.bind_context(**extra).info(message)

    def error(self, message, **extra):
        """Structured error log with extra data"""
        return self.bind_context(**extra).error(message)

# Usage Patterns
structured_logger = StructuredLogger()

# Initialize logger with context
logger = logger.patch(lambda record: record["extra"].update(
    environment=os.getenv("ENVIRONMENT", "development"),
    version="1.0.0"
))

# Request-specific logging
request_logger = structured_logger.bind_context(
    user_id="123",
    correlation_id="req-12345"
)

request_logger.info("Processing request")
logger.info("Background process started")
```

## 3. Exception Handling Best Practices

### \*\*3.1 Production Exception Configuration

```python
# Production configuration with enhanced backtrace
logger.add("logs/errors.log",
    backtrace=True,
    diagnose=False,  # Disable diagnose in production
    level="ERROR")

# Exception handling patterns
try:
    result = 1/0
except ZeroDivisionError:
    logger.exception("Division by zero - production safe")

# Catch decorator with production configuration
@logger.catch(diagnose=False)  # Disable diagnose in production
def risky_function():
    return 1/0
```

### \*\*3.2 Context managers for scoped logging

```python
# Context managers with contextual data
with logger.contextualize(user_id=123):
    logger.info("User logged in")
    # All logs will include user_id=123
    logger.info("Processing user operation")
```

## 4. File Handler Configuration

### \*\*4.1 Rotation and Compression Patterns

```python
# File handler with rotation and compression
logger.add("app.log",
    rotation="500 MB",      # Rotate when file reaches 500 MB
    retention="30 days",   # Keep files for 30 days
    compression="zip",     # Compress old files
    level="DEBUG"
)

# Multiple handlers for different log types
logger.add("errors.log",
    rotation="100 MB",
    retention="90 days",
    diagnose=False,
    backtrace=True,
    level="ERROR"
)

# Secure file permissions
def secure_file_permissions(file, flags):
    return os.open(file, flags, 0o600)  # Secure permissions

logger.add("secure.log", opener=secure_file_permissions)
```

## 5. Migration from Standard Logging

### \*\*5.1 Migration Patterns

```python
# Replace basicConfig
import sys
from loguru import logger

# Replace: logging.basicConfig(**config)
logger.remove()
logger.add(sys.stdout, level="INFO")

# Replace: logging.getLogger("app")
# Use: logger directly

# Replace: logging.info("message")
# Use: logger.info("message")

# Replace: logging.exception("error")
# Use: logger.exception("error")
```

### \*\*5.2 Extra Parameter Migration

```python
# Replace: logging.info("User {id}", extra={"id": user_id})
# Use: logger.bind(id=user_id).info("User logged in")

# Replace: logger.info("User created", extra={"user_id": user_id})
# Use: logger.bind(user_id=user_id).info("User created")
```

## 6. Testing Strategies

### \*\*6.1 Logot Testing Patterns

```python
# pytest configuration
# pyproject.toml
[tool.pytest.ini_options]
logot_capturer = "logot.loguru.LoguruCapturer"

# Test example
import pytest
from logot import Logot, logged

def test_user_operation(logot: Logot) -> None:
    user_operation()
    logot.assert_logged(logged.info("User created successfully"))
```

## 7. Performance Optimization

### \*\*7.1 Lazy Evaluation

```python
# Lazy evaluation for expensive operations
expensive_data = calculate_expensive_operation()
logger.opt(lazy=True).debug("Expensive operation result: {}", expensive_data)

# Filter functions for selective logging
def info_or_higher(record):
    return record["level"].no >= 30

logger.add("logs/all.log", filter=info_or_higher)
```

## 8. Security Best Practices

### \*\*8.1 Log Injection Protection

```python
# Log injection protection
def sanitize_input(text):
    return text.replace("\n", " ").replace("\r", " ")

username = sanitize_input(user_input)
logger.info("User {} logged in", username)

# NEVER log secrets
# Use environment variables
# Set secure file permissions
def secure_opener(file, flags):
    return os.open(file, flags, 0o600)

logger.add("secure.log", opener=secure_opener)
```

## 9. Docker Compose Configuration

### \*\*9.1 Docker Logging Configuration

````yaml
services:
  app:
    image: your-app:latest
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "3"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - LOGURU_FORMAT={time} [{level] {extra[request_id]} - {message}
      - LOGURU_DIAGNOSE=NO

## 10. Complete Scripts

### **10.1 Development Scripts
```bash
# scripts/dev.sh
#!/bin/bash
set -euo pipefail

# Start development server with auto-reload
uvicorn src/main:app --reload

# Run all tests
./scripts/test.sh

# Lint and format
./scripts/lint.sh
````

### \*\*10.2 Testing Scripts

```bash
# scripts/test.sh
#!/bin/bash
set -euo pipefail

# Run all unit and integration tests
pytest -q

# Run coverage
coverage run -m pytest
coverage html
```

## 11. Key Recommendations for MAIE Project

### \*\*11.1 Issues Addressed

1. ✅ **Replace mixed logging frameworks** with official best practices
2. ✅ **Implement environment-aware configuration**
3. ✅ **Implement correlation ID patterns**
4. ✅ **Configure multiple handlers** for different log destinations
5. ✅ **Implement structured logging** with JSON serialization
6. ✅ **Configure Docker deployment** patterns
7. ✅ **Implement signal handling** for graceful shutdown
8. ✅ **Configure comprehensive testing** strategies
9. ✅ **Implement log injection protection**
10. ✅ **Disable diagnose=True** in production

### \*\*11.2 Critical Recommendations

- **Docker deployments should log to stdout** - let container orchestration handle log aggregation
- **Use signal handling for graceful shutdown** - use `logger.complete()` before container shutdown
- **Use `logot` library for testing** - official recommendation for testing Loguru implementations
- **Disable `diagnose=True` in production** - security best practices
- **Use environment variables for configuration** - security best practices
- **Use lazy evaluation for expensive operations**

## 12. MCP Research Summary

### **Key Findings**

- Official Documentation: [loguru.readthedocs.io](https://loguru.readthedocs.io)
- GitHub Repository: [github.com/Delgan/loguru](https://github.com/Delgan/loguru)
- Docker Best Practices: [Docker Logging Best Practices](https://www.docker.com/blog/best-practices-for-logging-in-docker-containers/)

### **Production Implementation Recommendations**

- **Replace current mixed logging frameworks** with official best practices
- \*\*Implement
