import mlflow
import mlflow.tensorflow
from cnnClassifier.logger import configure_logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
import tensorflow as tf
import yaml
import os
from pathlib import Path
import itertools
from datetime import datetime
import json
import shutil

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

logger = configure_logger(logger_name=__name__)

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class MultiParamExperimentRunner:
    def __init__(self, base_params_file="params.yaml"):
        self.base_params_file = base_params_file
        self.results = []
        self.experiment_name = f"multi_param_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_available = False
        
        # Test MLflow connection
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            logger.info(f"MLflow connected successfully! Found {len(experiments)} experiments")
            self.mlflow_available = True
            
            # Create experiment if MLflow is available
            try:
                exp_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {exp_id})")
            except Exception as e:
                logger.info(f"Experiment {self.experiment_name} already exists or error: {e}")
                pass  # Experiment already exists
                
        except Exception as e:
            logger.warning(f"MLflow connection failed: {e}")
            logger.info("Continuing without MLflow - results will be saved to JSON files only")
            self.mlflow_available = False
        
    def load_base_params(self):
        """Load base parameters from yaml file"""
        with open(self.base_params_file, 'r') as file:
            return yaml.safe_load(file)
    
    def create_param_combinations(self, param_variations):
        """
        Create all combinations of parameters to test
        
        Args:
            param_variations (dict): Dictionary with parameter names as keys and lists of values to test
            
        Example:
            param_variations = {
                'LEARNING_RATE': [0.01, 0.001, 0.0001],
                'MODEL_NAME': ['VGG16', 'VGG19', 'MobileNetV3Large'],
                'BATCH_SIZE': [16, 32]
            }
        """
        keys = param_variations.keys()
        values = param_variations.values()
        combinations = list(itertools.product(*values))
        
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
            
        return param_combinations
    
    def create_temp_params_file(self, base_params, param_overrides, experiment_id):
        """Create a temporary params file with overridden values"""
        temp_params = base_params.copy()
        temp_params.update(param_overrides)
        
        temp_params_file = f"params_temp_{experiment_id}.yaml"
        
        with open(temp_params_file, 'w') as file:
            yaml.dump(temp_params, file, default_flow_style=False)
            
        return temp_params_file
    
    def run_single_experiment(self, param_overrides, experiment_id):
        """Run a single experiment with given parameters"""
        logger.info(f"Starting experiment {experiment_id} with params: {param_overrides}")
        
        # Load base parameters
        base_params = self.load_base_params()
        
        # Create temporary params file
        temp_params_file = self.create_temp_params_file(base_params, param_overrides, experiment_id)
        
        # Update the params.yaml temporarily
        original_params_backup = "params_backup.yaml"
        os.rename("params.yaml", original_params_backup)
        os.rename(temp_params_file, "params.yaml")
        
        experiment_results = {
            'experiment_id': experiment_id,
            'parameters': param_overrides,
            'status': 'running',
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Try to start MLflow run, but continue even if it fails
            mlflow_success = False
            try:
                with mlflow.start_run(run_name=f"exp_{experiment_id}"):
                    mlflow_success = True
                    logger.info(f"Started MLflow run for experiment {experiment_id}")
                    
                    # Log all base parameters first
                    base_params = self.load_base_params()
                    logger.info(f"Logging {len(base_params)} base parameters")
                    for key, value in base_params.items():
                        try:
                            mlflow.log_param(f"base_{key}", value)
                        except Exception as e:
                            logger.warning(f"Failed to log base param {key}: {e}")
                    
                    # Log override parameters
                    logger.info(f"Logging {len(param_overrides)} override parameters")
                    for key, value in param_overrides.items():
                        try:
                            mlflow.log_param(key, value)
                        except Exception as e:
                            logger.warning(f"Failed to log override param {key}: {e}")
                    
                    # Log experiment metadata
                    mlflow.log_param("experiment_id", experiment_id)
                    mlflow.log_param("experiment_name", self.experiment_name)
                    mlflow.log_param("start_time", experiment_results['start_time'])
                    
                    # Run the pipeline stages
                    self.run_prepare_base_model()
                    training_metrics = self.run_model_training()
                    evaluation_metrics = self.run_model_evaluation()
                    
                    # Log metrics to MLflow
                    self._log_metrics_to_mlflow(training_metrics, evaluation_metrics)
                    
                    # Log artifacts
                    self.log_artifacts(experiment_id)
                    
                    # Log success metrics
                    mlflow.log_metric("experiment_success", 1)
                    mlflow.log_metric("training_success", 1 if training_metrics else 0)
                    mlflow.log_metric("evaluation_success", 1 if evaluation_metrics else 0)
                    
            except Exception as mlflow_error:
                logger.warning(f"MLflow logging failed, continuing without it: {mlflow_error}")
                mlflow_success = False
                
                # Run pipeline without MLflow
                self.run_prepare_base_model()
                training_metrics = self.run_model_training()
                evaluation_metrics = self.run_model_evaluation()
            
            # Always save results to JSON regardless of MLflow status
            experiment_results.update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics,
                'mlflow_logged': mlflow_success
            })
            
            logger.info(f"Experiment {experiment_id} completed successfully (MLflow: {'✅' if mlflow_success else '❌'})")
                
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Log failure to MLflow if run is still active
            try:
                if mlflow.active_run():
                    mlflow.log_metric("experiment_success", 0)
                    mlflow.log_param("error_message", str(e))
                    mlflow.log_param("error_type", type(e).__name__)
                    logger.info("Logged error information to MLflow")
            except Exception as mlflow_error:
                logger.error(f"Failed to log error to MLflow: {mlflow_error}")
            
            experiment_results.update({
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'end_time': datetime.now().isoformat()
            })
            
        finally:
            # Restore original params file
            os.rename("params.yaml", f"params_used_{experiment_id}.yaml")
            os.rename(original_params_backup, "params.yaml")
            
            # Clean up GPU memory
            tf.keras.backend.clear_session()
            
        return experiment_results
    
    def _log_metrics_to_mlflow(self, training_metrics, evaluation_metrics):
        """Helper method to log metrics to MLflow"""
        try:
            # Log training metrics
            if training_metrics:
                logger.info(f"Logging {len(training_metrics)} training metrics")
                for key, value in training_metrics.items():
                    try:
                        mlflow.log_metric(f"train_{key}", float(value))
                        logger.debug(f"Logged training metric {key}={value}")
                    except (ValueError, TypeError):
                        mlflow.log_param(f"train_{key}", str(value))
                        logger.debug(f"Logged training param {key}={value}")
                    except Exception as e:
                        logger.warning(f"Failed to log training metric {key}: {e}")
            
            # Log evaluation metrics
            if evaluation_metrics:
                logger.info(f"Logging {len(evaluation_metrics)} evaluation metrics")
                for key, value in evaluation_metrics.items():
                    try:
                        mlflow.log_metric(f"eval_{key}", float(value))
                        logger.debug(f"Logged evaluation metric {key}={value}")
                    except (ValueError, TypeError):
                        mlflow.log_param(f"eval_{key}", str(value))
                        logger.debug(f"Logged evaluation param {key}={value}")
                    except Exception as e:
                        logger.warning(f"Failed to log evaluation metric {key}: {e}")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def run_data_ingestion(self):
        """Run data ingestion stage"""
        try:
            logger.info(">>>>> stage Data Ingestion started <<<<<<<")
            data_ingestion_pipeline = DataIngestionPipeline()
            data_ingestion_pipeline.main()
            logger.info(">>>>> stage Data Ingestion completed <<<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e
    
    def run_prepare_base_model(self):
        """Run prepare base model stage"""
        try:
            logger.info(">>>>> stage Prepare Base Model started <<<<<<<")
            prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
            prepare_base_model_pipeline.main()
            logger.info(">>>>> stage Prepare Base Model completed <<<<<<<")
        except Exception as e:
            logger.exception(e)
            raise e
    
    def run_model_training(self):
        """Run model training stage and return metrics"""
        try:
            logger.info(">>>>> stage Model Training started <<<<<<<")
            model_training_pipeline = ModelTrainingPipeline()
            model_training_pipeline.main()
            logger.info(">>>>> stage Model Training completed <<<<<<<")
            
            # Extract training metrics from model history if available
            training_metrics = self.extract_training_metrics()
            
            # Log model artifacts
            model_path = "artifacts/training/model.keras"
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path)
                    mlflow.tensorflow.log_model(
                        model,
                        "model",
                        keras_model_kwargs={"save_format": "keras"}
                    )
                    logger.info("Model logged to MLflow successfully")
                except Exception as model_log_error:
                    logger.warning(f"Failed to log model to MLflow: {model_log_error}")
                    # Log the model as an artifact instead
                    try:
                        mlflow.log_artifact(model_path, "models")
                        logger.info("Model logged as artifact instead")
                    except Exception as artifact_error:
                        logger.warning(f"Failed to log model as artifact: {artifact_error}")
            
            return training_metrics
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def run_model_evaluation(self):
        """Run model evaluation stage and return metrics"""
        try:
            logger.info(">>>>> stage Model Evaluation started <<<<<<<")
            
            # Temporarily disable the MLflow logging in the evaluation pipeline
            # since we're already in a MLflow run context
            import os
            original_mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
            
            # Set environment variable to disable MLflow in evaluation pipeline
            os.environ['DISABLE_MLFLOW_EVALUATION'] = 'true'
            
            evaluation_pipeline = EvaluationPipeline()
            evaluation_pipeline.main()
            
            # Restore original environment
            if original_mlflow_uri:
                os.environ['MLFLOW_TRACKING_URI'] = original_mlflow_uri
            else:
                os.environ.pop('MLFLOW_TRACKING_URI', None)
            os.environ.pop('DISABLE_MLFLOW_EVALUATION', None)
            
            logger.info(">>>>> stage Model Evaluation completed <<<<<<<")
            
            # Extract evaluation metrics
            evaluation_metrics = self.extract_evaluation_metrics()
            
            return evaluation_metrics
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def extract_training_metrics(self):
        """Extract training metrics from logs or saved files"""
        metrics = {}
        
        try:
            # Try to read training logs or history
            log_file = "logs/running_logs.log"
            if os.path.exists(log_file):
                logger.info(f"Reading training logs from {log_file}")
                # Parse logs for metrics (simplified example)
                with open(log_file, 'r') as f:
                    content = f.read()
                    # You can add more sophisticated parsing here
                    metrics['training_completed'] = 1
                    metrics['log_size'] = len(content)
            else:
                logger.warning(f"Log file {log_file} not found")
                metrics['log_file_missing'] = 1
            
            # Check if model exists and get basic info
            model_path = "artifacts/training/model.keras"
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                metrics['model_params'] = model.count_params()
                metrics['model_layers'] = len(model.layers)
                metrics['model_exists'] = 1
                logger.info(f"Model info: {metrics['model_params']} params, {metrics['model_layers']} layers")
            else:
                logger.warning(f"Model file {model_path} not found")
                metrics['model_exists'] = 0
                
        except Exception as e:
            logger.error(f"Could not extract training metrics: {e}")
            metrics['training_completed'] = 0
            metrics['extraction_error'] = str(e)
        
        logger.info(f"Extracted training metrics: {metrics}")
        return metrics
    
    def extract_evaluation_metrics(self):
        """Extract evaluation metrics from scores.json or other files"""
        metrics = {}
        
        try:
            # Try to read evaluation scores
            scores_file = "scores.json"
            if os.path.exists(scores_file):
                logger.info(f"Reading evaluation metrics from {scores_file}")
                with open(scores_file, 'r') as f:
                    scores = json.load(f)
                    logger.info(f"Found scores: {scores}")
                    metrics.update(scores)
                    metrics['evaluation_completed'] = 1
            else:
                logger.warning(f"Scores file {scores_file} not found")
                # Default metrics if file doesn't exist
                metrics['evaluation_completed'] = 0
                metrics['scores_file_missing'] = 1
            
            # NOVA FUNCIONALIDADE: Análise básica de activation rates
            try:
                activation_metrics = self._analyze_activation_rates_basic()
                metrics.update(activation_metrics)
            except Exception as e:
                logger.warning(f"Could not analyze activation rates: {e}")
                
        except Exception as e:
            logger.error(f"Could not extract evaluation metrics: {e}")
            metrics['evaluation_completed'] = 0
            metrics['extraction_error'] = str(e)
        
        logger.info(f"Extracted evaluation metrics: {metrics}")
        return metrics
    
    def _analyze_activation_rates_basic(self):
        """Análise básica de activation rates (simplificada para não atrasar experimentos)"""
        try:
            model_path = "artifacts/training/model.keras"
            if not os.path.exists(model_path):
                return {'activation_analysis': 'model_not_found'}
            
            # Carregar modelo e analisar arquitetura
            model = tf.keras.models.load_model(model_path)
            
            # Contar camadas e tipos
            dense_layers = 0
            dropout_layers = 0
            total_layers = len(model.layers)
            
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layers += 1
                elif isinstance(layer, tf.keras.layers.Dropout):
                    dropout_layers += 1
            
            return {
                'activation_total_layers': total_layers,
                'activation_dense_layers': dense_layers,
                'activation_dropout_layers': dropout_layers,
                'activation_architecture_score': dense_layers / max(1, total_layers),
                'activation_analysis': 'basic_completed'
            }
            
        except Exception as e:
            return {'activation_analysis': f'error_{str(e)[:50]}'}  # Truncar erro
    
    def log_artifacts(self, experiment_id):
        """Log important artifacts to MLflow"""
        try:
            # Log model files as artifacts (safer than model logging)
            model_files = [
                "artifacts/training/model.keras",
                "artifacts/training/model.h5"  # fallback if .h5 exists
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        mlflow.log_artifact(model_file, "models")
                        logger.info(f"Model file {model_file} logged as artifact")
                        break  # Only log one model file
                    except Exception as e:
                        logger.warning(f"Failed to log model file {model_file}: {e}")
            
            # Log configuration files
            config_file = f"params_used_{experiment_id}.yaml"
            if os.path.exists(config_file):
                try:
                    mlflow.log_artifact(config_file, "config")
                    logger.info(f"Config file {config_file} logged")
                except Exception as e:
                    logger.warning(f"Failed to log config file: {e}")
            
            # Log scores if available
            if os.path.exists("scores.json"):
                try:
                    mlflow.log_artifact("scores.json", "metrics")
                    logger.info("Scores file logged")
                except Exception as e:
                    logger.warning(f"Failed to log scores file: {e}")
            
            # Log training logs
            if os.path.exists("logs/running_logs.log"):
                try:
                    mlflow.log_artifact("logs/running_logs.log", "logs")
                    logger.info("Training logs logged")
                except Exception as e:
                    logger.warning(f"Failed to log training logs: {e}")
                
            logger.info("Artifacts logging completed")
            
        except Exception as e:
            logger.error(f"Could not log artifacts: {e}")
            # Don't raise the exception to avoid failing the entire experiment
    
    def run_experiments(self, param_variations):
        """Run all parameter combination experiments"""
        combinations = self.create_param_combinations(param_variations)
        
        logger.info(f"Starting multi-parameter experiment with {len(combinations)} combinations")
        logger.info(f"MLflow status: {'✅ Available' if self.mlflow_available else '❌ Not available (JSON only)'}")
        
        # Set MLflow experiment only if available
        if self.mlflow_available:
            try:
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                logger.warning(f"Failed to set MLflow experiment: {e}")
                self.mlflow_available = False
        
        for i, param_combo in enumerate(combinations):
            experiment_id = f"{i+1:03d}"
            
            try:
                result = self.run_single_experiment(param_combo, experiment_id)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
                # Show progress
                success_status = "✅" if result['status'] == 'completed' else "❌"
                mlflow_status = "✅" if result.get('mlflow_logged', False) else "❌"
                logger.info(f"Progress: {i+1}/{len(combinations)} | Exp: {success_status} | MLflow: {mlflow_status}")
                
            except Exception as e:
                logger.error(f"Failed to run experiment {experiment_id}: {str(e)}")
                continue
        
        # Final results summary
        self.generate_summary_report()
        
        return self.results
    
    def save_results(self):
        """Save experiment results to JSON file"""
        results_file = f"experiment_results_{self.experiment_name}.json"
        with open(results_file, 'w') as file:
            json.dump(self.results, file, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of all experiments"""
        successful_experiments = [r for r in self.results if r['status'] == 'completed']
        failed_experiments = [r for r in self.results if r['status'] == 'failed']
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_experiments),
            'failed_experiments': len(failed_experiments),
            'success_rate': len(successful_experiments) / len(self.results) * 100 if self.results else 0
        }
        
        # Save summary
        summary_file = f"experiment_summary_{self.experiment_name}.json"
        with open(summary_file, 'w') as file:
            json.dump(summary, file, indent=2, default=str)
        
        logger.info(f"Experiment Summary:")
        logger.info(f"Total experiments: {summary['total_experiments']}")
        logger.info(f"Successful: {summary['successful_experiments']}")
        logger.info(f"Failed: {summary['failed_experiments']}")
        logger.info(f"Success rate: {summary['success_rate']:.2f}%")


def main():
    """Main function to run multi-parameter experiments"""
    
    # Define parameter variations to test - TESTE DE CORREÇÃO DO DROPOUT
    param_variations = {
        # 'LEARNING_RATE': [0.01, 0.001, 0.005],
        'LEARNING_RATE': [0.001],
        # 'MODEL_NAME': ['MobileNetV3Large'],
        'MODEL_NAME': ['VGG19'],

        # 'MODEL_NAME': ['VGG16','VGG19'],
        'BATCH_SIZE': [16],
        'EPOCHS': [100],  # Teste bem rápido
        'DROPOUT_RATE': [0.0]  # Comparar sem dropout vs com dropout
    }
    
    # Alternative: smaller test for debugging
    # param_variations = {
    #     'LEARNING_RATE': [0.01, 0.001],
    #     'MODEL_NAME': ['VGG16', 'MobileNetV3Large']
    # }
    
    # Create and run experiments
    runner = MultiParamExperimentRunner()
    
    logger.info(f"Starting experiment: {runner.experiment_name}")
    logger.info(f"Parameter variations: {param_variations}")
    logger.info(f"Total combinations to test: {len(list(itertools.product(*param_variations.values())))}")
    logger.info(f"MLflow status: {'✅ Available' if runner.mlflow_available else '❌ Not available (results saved to JSON only)'}")
    
    if runner.mlflow_available:
        logger.info(f"View results in MLflow UI: http://127.0.0.1:5001")
    else:
        logger.info("Results will be saved to JSON files in the current directory")
    
    results = runner.run_experiments(param_variations)
    
    logger.info("All experiments completed!")
    logger.info(f"Results saved in: experiment_results_{runner.experiment_name}.json")
    
    if runner.mlflow_available:
        logger.info(f"View results in MLflow UI: http://127.0.0.1:5001")
    
    return results


if __name__ == "__main__":
    results = main()
