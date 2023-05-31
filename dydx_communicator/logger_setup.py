from colorama import Fore, Style
import logging
import os



class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    def format(self, record):
        if record.levelno == logging.ERROR:
            return f"{Fore.RED}{super().format(record)}{Style.RESET_ALL}"
        else:
            return super().format(record)



def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "system_log")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, f"{name}.log")

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and attach it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
