# -*- coding: utf-8 -*-
import logging

from rich.logging import RichHandler


def get_logger(name: str = "rich", level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(name)s: %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


logger = get_logger("utils.py")
