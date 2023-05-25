import logging
import os


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Get the directory where this logger_setup.py file is located
    log_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "system_log")

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, f"{name}.log")

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
