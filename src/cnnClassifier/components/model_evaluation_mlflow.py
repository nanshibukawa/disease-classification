import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _create_valid_generator(self):
        """
        Cria gerador para avaliação usando dados de teste (se disponível) ou validação
        """
        # Verificar se existe conjunto de teste separado
        split_test_dir = Path("artifacts/data_split/test")
        split_val_dir = Path("artifacts/data_split/validation")
        
        if split_test_dir.exists():
            # Usar conjunto de teste (ideal para avaliação final)
            target_dir = split_test_dir
            datagenerator_kwargs = dict(rescale=1.0 / 255.0)
            print("Usando conjunto de TESTE para avaliação final")
        elif split_val_dir.exists():
            # Usar conjunto de validação separado
            target_dir = split_val_dir
            datagenerator_kwargs = dict(rescale=1.0 / 255.0)
            print("Usando conjunto de VALIDAÇÃO separado para avaliação")
        else:
            # Fallback para divisão automática
            target_dir = self.config.training_data
            datagenerator_kwargs = dict(
                rescale=1.0 / 255.0,
                validation_split=0.3
            )
            print("Usando divisão automática para avaliação (fallback)")

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        if split_test_dir.exists() or split_val_dir.exists():
            # Usar diretório específico (sem subset)
            self._valid_generator = valid_datagenerator.flow_from_directory(
                directory=target_dir,
                shuffle=False,
                **dataflow_kwargs
            )
        else:
            # Usar subset validation (divisão automática)
            self._valid_generator = valid_datagenerator.flow_from_directory(
                directory=target_dir,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    # def evaluation(self):
    #     self.model = self.load_model(self.config.path_of_model)
    #     self._create_valid_generator()
    #     self.score = self.model.evaluate(self._valid_generator)

    #     # Get predictions and true labels
    #     y_pred_probs = self.model.predict(self._valid_generator)
    #     import numpy as np
    #     from sklearn.metrics import precision_score, recall_score, f1_score
    #     y_pred = np.argmax(y_pred_probs, axis=1)
    #     y_true = self._valid_generator.classes

    #     # Calculate metrics
    #     precision = precision_score(y_true, y_pred, average="weighted")
    #     recall = recall_score(y_true, y_pred, average="weighted")
    #     f1 = f1_score(y_true, y_pred, average="weighted")

    #     self.extra_metrics = {
    #         "precision": float(precision),
    #         "recall": float(recall),
    #         "f1_score": float(f1)
    #     }
    #     self.save_score()

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._create_valid_generator()
        self.score = self.model.evaluate(self._valid_generator)

        # Get predictions and true labels
        y_pred_probs = self.model.predict(self._valid_generator)
        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self._valid_generator.classes

        # Calculate metrics
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        self.extra_metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

        # Print F1-score as main metric in the terminal
        print("================= EVALUATION METRICS =================")
        print(f"F1-score (weighted): {f1:.4f}   <=== MAIN METRIC")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"Loss: {self.score[0]:.4f}")
        print(f"Accuracy: {self.score[1]:.4f}")
        print("======================================================")

        self.save_score()

    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
        }
        if hasattr(self, "extra_metrics"):
            scores.update(self.extra_metrics)
        save_json(
            path=Path("scores.json"),
            data=scores
        )

    def log_into_mlflow(self):
        # Check if MLflow logging should be disabled (when running in multi-param mode)
        import os
        if os.environ.get('DISABLE_MLFLOW_EVALUATION') == 'true':
            print("MLflow logging disabled for evaluation pipeline (already in MLflow run context)")
            return
            
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Check if there's already an active run
        active_run = mlflow.active_run()
        if active_run:
            print(f"Using existing MLflow run: {active_run.info.run_id}")
            # Log to existing run instead of creating new one
            mlflow.log_params(self.config.all_params)
            # Log main metrics
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
            })
            # Log extra metrics if available
            if hasattr(self, "extra_metrics"):
                mlflow.log_metrics(self.extra_metrics)

            # --- LOGGING MODEL AS ARTIFACT ---
            local_model_path = Path("VGG19Model.keras")
            self.model.save(local_model_path)
            print(f"Model temporarily saved to: {local_model_path}")
            mlflow.log_artifact(str(local_model_path), artifact_path="model.keras")
            if local_model_path.exists():
                os.remove(local_model_path)
        else:
            # Create new run only if no active run exists
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                # Log main metrics
                mlflow.log_metrics({
                    "loss": self.score[0],
                    "accuracy": self.score[1],
                })
                # Log extra metrics if available
                if hasattr(self, "extra_metrics"):
                    mlflow.log_metrics(self.extra_metrics)

                # --- LOGGING MODEL AS ARTIFACT ---
                local_model_path = Path("VGG19Model.keras")
                self.model.save(local_model_path)
                print(f"Model temporarily saved to: {local_model_path}")
                mlflow.log_artifact(str(local_model_path), artifact_path="model.keras")
                if local_model_path.exists():
                    os.remove(local_model_path)
                print(f"Cleaned up local model file: {local_model_path}")
