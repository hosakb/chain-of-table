import logging
import os
from typing import Dict

_LOGGER_CACHE: Dict[str, logging.Logger] = {}


def get_logger(
    table_name: str, module_name: str, logs_dir: str = "./logs"
) -> logging.Logger:
    table_name = table_name.replace("/", "-")

    logger_key = f"{table_name}_{module_name}"
    if logger_key in _LOGGER_CACHE:
        return _LOGGER_CACHE[logger_key]

    # Create table-specific directory
    table_dir = os.path.join(logs_dir, table_name)
    os.makedirs(table_dir, exist_ok=True)

    logger = logging.getLogger(logger_key)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Avoid duplicate logs

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Table-specific log file (overwrite each run)
    table_file_path = os.path.join(table_dir, f"{module_name}.log")
    table_file_handler = logging.FileHandler(
        table_file_path, mode="a", encoding="utf-8"
    )
    table_file_handler.setLevel(logging.DEBUG)
    table_file_handler.setFormatter(formatter)
    logger.addHandler(table_file_handler)

    # Global log file for the same module (overwrite each run)
    global_dir = os.path.join(logs_dir, "GLOBAL")
    os.makedirs(global_dir, exist_ok=True)

    global_file_path = os.path.join(global_dir, f"{module_name}.log")
    global_file_handler = logging.FileHandler(
        global_file_path, mode="a", encoding="utf-8"
    )
    global_file_handler.setLevel(logging.DEBUG)
    global_file_handler.setFormatter(formatter)
    logger.addHandler(global_file_handler)

    _LOGGER_CACHE[logger_key] = logger
    return logger