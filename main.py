from cnnClassifier.logger import configure_logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
import tensorflow as tf
tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


logger = configure_logger(logger_name =__name__)

# logger = logger()

# STAGE_NAME = "Data Ingestion stage"

# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
#     data_ingestion_pipeline = DataIngestionPipeline()
#     data_ingestion_pipeline.main()
#     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
    prepare_base_model_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


import gc
tf.keras.backend.clear_session()
gc.collect()