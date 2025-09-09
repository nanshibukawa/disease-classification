import mlflow
from cnnClassifier.logger import configure_logger
from cnnClassifier.pipeline.stage_00_data_splitting import DataSplittingPipeline
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
import tensorflow as tf
import yaml
import os
from datetime import datetime

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


logger = configure_logger(logger_name =__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configuration for multi-parameter testing
MULTI_PARAM_MODE = os.getenv("MULTI_PARAM_MODE", "False").lower() == "true"
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def update_params_file(param_overrides):
    """Update params.yaml with new parameter values"""
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)
    
    params.update(param_overrides)
    
    with open("params.yaml", 'w') as file:
        yaml.dump(params, file, default_flow_style=False)

def run_single_pipeline_stage(stage_name, pipeline_func):
    """Run a single pipeline stage with error handling"""
    try:
        logger.info(f">>>>> stage {stage_name} started <<<<<<<")
        pipeline_func()
        logger.info(f">>>>> stage {stage_name} completed <<<<<<<\n\nx==========x")
        return True
    except Exception as e:
        logger.exception(e)
        if not MULTI_PARAM_MODE:
            raise e
        return False

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    stages_success = []
    
    # Data Ingestion (commented out as in original)
    # stages_success.append(run_single_pipeline_stage(
    #     "Data Ingestion stage",
    #     lambda: DataIngestionPipeline().main()
    # ))
    
    # Data Splitting (execute only if split doesn't exist)
    from pathlib import Path
    if not Path("artifacts/data_split").exists():
        stages_success.append(run_single_pipeline_stage(
            "Data Splitting",
            lambda: DataSplittingPipeline().main()
        ))
    else:
        logger.info("Data split already exists, skipping data splitting stage")
    
    # Prepare Base Model
    stages_success.append(run_single_pipeline_stage(
        "Prepare Base Model",
        lambda: PrepareBaseModelTrainingPipeline().main()
    ))
    
    # Model Training
    stages_success.append(run_single_pipeline_stage(
        "Model Training",
        lambda: ModelTrainingPipeline().main()
    ))
    
    # Model Evaluation
    stages_success.append(run_single_pipeline_stage(
        "Model Evaluation",
        lambda: EvaluationPipeline().main()
    ))
    
    return all(stages_success)

def run_multi_param_experiments():
    """Run experiments with multiple parameter combinations"""
    # Define parameter variations
    param_variations = [
        {'LEARNING_RATE': 0.01, 'MODEL_NAME': 'VGG16', 'BATCH_SIZE': 16},
        {'LEARNING_RATE': 0.001, 'MODEL_NAME': 'VGG16', 'BATCH_SIZE': 32},
        {'LEARNING_RATE': 0.01, 'MODEL_NAME': 'VGG19', 'BATCH_SIZE': 16},
        {'LEARNING_RATE': 0.001, 'MODEL_NAME': 'MobileNetV3Large', 'BATCH_SIZE': 16},
    ]
    
    # Backup original params
    with open("params.yaml", 'r') as file:
        original_params = yaml.safe_load(file)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []
    
    for i, param_combo in enumerate(param_variations):
        experiment_id = f"exp_{i+1:03d}"
        logger.info(f"Starting experiment {experiment_id} with params: {param_combo}")
        
        try:
            with mlflow.start_run(run_name=experiment_id):
                # Update parameters
                update_params_file(param_combo)
                
                # Log parameters to MLflow
                for key, value in param_combo.items():
                    mlflow.log_param(key, value)
                
                # Run pipeline
                success = run_complete_pipeline()
                
                # Log success metric
                mlflow.log_metric("pipeline_success", 1 if success else 0)
                
                results.append({
                    'experiment_id': experiment_id,
                    'parameters': param_combo,
                    'success': success
                })
                
                logger.info(f"Experiment {experiment_id} {'completed' if success else 'failed'}")
                
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed with error: {str(e)}")
            results.append({
                'experiment_id': experiment_id,
                'parameters': param_combo,
                'success': False,
                'error': str(e)
            })
        
        finally:
            # Clean up GPU memory between experiments
            tf.keras.backend.clear_session()
    
    # Restore original parameters
    with open("params.yaml", 'w') as file:
        yaml.dump(original_params, file, default_flow_style=False)
    
    # Log summary
    successful_experiments = sum(1 for r in results if r['success'])
    logger.info(f"Multi-parameter experiments completed:")
    logger.info(f"Total: {len(results)}, Successful: {successful_experiments}, Failed: {len(results) - successful_experiments}")
    
    return results


# STAGE_NAME = "Data Ingestion stage"

# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
#     data_ingestion_pipeline = DataIngestionPipeline()
#     data_ingestion_pipeline.main()
#     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


if __name__ == "__main__":
    if MULTI_PARAM_MODE:
        logger.info("Running in multi-parameter mode")
        results = run_multi_param_experiments()
    else:
        logger.info("Running single pipeline")
        success = run_complete_pipeline()
        if success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline failed!")

# Original single-run code (commented out for reference)
"""
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


STAGE_NAME = "Model Evaluation"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    evaluation_pipeline = EvaluationPipeline()
    evaluation_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


import gc
tf.keras.backend.clear_session()
gc.collect()
"""