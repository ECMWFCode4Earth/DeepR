import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance configured with an INFO level, logging to stdout.

    The logger is configured with a handler and a formatter already set up.
    The handler sends log records to the standard output (stdout).

    Parameters
    ----------
    name : str
        Name for the logger.

    Returns
    -------
    logger : logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel("INFO")
    logger.propagate = False
    return logger
