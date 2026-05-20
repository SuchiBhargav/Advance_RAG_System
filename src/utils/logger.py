"""
Comprehensive logging setup with structured logging, rotation, and monitoring.
Provides consistent logging across the application with proper formatting.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
import json
from datetime import datetime

from config.settings import settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in structured JSON format.
    Useful for log aggregation and analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        # Extra fields are added directly to the record object, not as a dict
        for key, value in record.__dict__.items():
            # Skip standard LogRecord attributes
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'getMessage', 'taskName']:
                log_data[key] = value
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    Makes logs more readable during development.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Color the level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    use_json: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        use_json: Whether to use JSON formatting for file logs
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level from settings or parameter
    level = log_level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if settings.ENVIRONMENT == "production":
        # Use simple format in production
        console_formatter = logging.Formatter(settings.LOG_FORMAT)
    else:
        # Use colored format in development
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file or settings.LOG_FILE:
        log_path = Path(log_file or settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (rotates when file reaches 10MB)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Use JSON formatter for file logs if requested
        if use_json:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                settings.LOG_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return setup_logger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds extra context to log messages.
    Useful for adding request IDs, user IDs, etc.
    """
    
    def process(self, msg, kwargs):
        """Add extra fields to log record."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add adapter's extra fields
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs


def get_logger_with_context(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with additional context fields.
    
    Args:
        name: Logger name
        **context: Additional context fields to include in logs
    
    Returns:
        LoggerAdapter with context
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)


# Create application-wide loggers
app_logger = get_logger("app")
api_logger = get_logger("api")
rag_logger = get_logger("rag")
db_logger = get_logger("database")
eval_logger = get_logger("evaluation")

# Made with Bob
