import logging
import os


def setup_logger(name):
    """Sets up a logger for a given module.

    Args:
        name (str): The name of the module.

    Returns:
        Logger: The logger object.
    """


    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_directory = "system_logs"  # specify your directory here
    if not os.path.exists(log_directory):  # check if directory exists
        os.makedirs(log_directory)  # if not, create it

    log_file_path = os.path.join(log_directory, f"{name}.log")

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    
    return logger
