"""
Error handling module for AC-OPF problems.

This module provides utilities for handling errors in power system operations.
"""

import functools
import logging
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class ErrorCodes(Enum):
    """Error codes for power system operations."""
    UNKNOWN_ERROR = 0
    POWER_FLOW_DIVERGED = 1
    INVALID_CASE_DATA = 2
    OPTIMIZATION_FAILED = 3
    CONSTRAINT_VIOLATION = 4

class PowerFlowError(Exception):
    """Exception raised for power flow calculation errors."""
    def __init__(self, message, code=ErrorCodes.POWER_FLOW_DIVERGED):
        self.message = message
        self.code = code
        super().__init__(self.message)

class ValidationError(Exception):
    """Exception raised for validation errors."""
    def __init__(self, message, code=ErrorCodes.CONSTRAINT_VIOLATION):
        self.message = message
        self.code = code
        super().__init__(self.message)

class ErrorContext:
    """Context for error handling."""
    def __init__(self, context_name):
        self.context_name = context_name
        
    def __enter__(self):
        logger.debug(f"Entering {self.context_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Error in {self.context_name}: {exc_val}")
        else:
            logger.debug(f"Exiting {self.context_name}")
        return False  # Do not suppress the exception

def retry(max_attempts=3, delay=1):
    """
    Retry decorator for functions that may fail temporarily.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts")
                        raise
                    logger.warning(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

def handle_errors(error_code=ErrorCodes.UNKNOWN_ERROR):
    """
    Error handling decorator.
    
    Args:
        error_code: Error code to use if an exception occurs
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                # You can add additional handling here
                raise
        return wrapper
    return decorator 