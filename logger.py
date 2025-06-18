from loguru import logger

def configure_logger(log_file="app.log", level="INFO"):
    logger.remove()  # Remove default handler
    logger.add(
        log_file,
        rotation="10 MB",
        retention="10 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

# Example usage:
# configure_logger()
# logger.info("Logger configured successfully.")