# from logger import configure_logger
from .logger import configure_logger

# Configure a default logger for the module
logger = configure_logger(logger_name="cnnClassifier")

# from pipeline.stage_01_data_ingestion import DataIngestionPipeline
# from pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# from pipeline.stage_03_model_training import ModelTrainingPipeline