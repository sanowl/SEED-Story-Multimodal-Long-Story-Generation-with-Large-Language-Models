import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name: str, log_file: str = 'app.log', level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
    name (str): Name of the logger.
    log_file (str): Path to the log file.
    level (int): Logging level.

    Returns:
    logging.Logger: Configured logger object.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    
    # Create formatters
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatters to handlers
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def log_exception(logger: logging.Logger, exc_info: tuple):
    """
    Log an exception with traceback.

    Args:
    logger (logging.Logger): Logger object.
    exc_info (tuple): Exception info from sys.exc_info().
    """
    logger.error("Exception occurred", exc_info=exc_info)