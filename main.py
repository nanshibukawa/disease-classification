from src.cnnClassifier.log.logger import configure_logger

logger = configure_logger()

def main():

    logger.debug("Logger configured successfully.")


if __name__ == "__main__":
    main()
