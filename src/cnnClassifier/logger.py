import os
import sys

from loguru import logger

def configure_logger(
    log_dir="logs",
    log_file="running_logs.log",
    level="INFO",
    log_format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level><bold>{level: <8}</bold></level> | "
        "<blue>{module}</blue>:<yellow>{function}</yellow>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
):
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file)

    logger.remove()  # Remove default handler
    logger.add(log_filepath, level=level, format=log_format, enqueue=True)
    logger.add(sys.stdout, level=level, format=log_format)
    return logger
