import logging
import sys

def setup_logger(log_to_file=False, log_file_name='app.log', log_level=logging.INFO):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_to_file:
        # File handler
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger