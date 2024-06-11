import logging
import os

from colorlog import ColoredFormatter

def init_logger(level: int, log_path: str, logger_name: str='extractor'):
    """
    Initialize the logger.
    :param level: The level of the logger.
    :param log_path: The path to the log file.
    :param logger_name: The name of the logger.
    :return: The logger.
    """
    logger = logging.getLogger(logger_name)
    formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(name)6s | %(log_color)s%(levelname)4s | %(log_color)s%(message)6s",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'yellow',
            'WARNING':  'green',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
    )

    file_formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(name)6s | %(log_color)s%(levelname)4s | %(log_color)s%(message)6s",
        reset=True,
        no_color=True,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log_file = os.path.abspath(path=log_path)
    open(log_file, 'w').close() # Clear the log file
    output_file_handler = logging.FileHandler(log_file, mode='a')
    output_file_handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    logger.addHandler(output_file_handler)
    logger.setLevel(level)

    return logger