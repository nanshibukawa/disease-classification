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

        datagenerator_kwargs = dict(
            rescale=1.0 / 255.0,
            validation_split=0.3
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
            )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self._valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._create_valid_generator()
        self.score = self.model.evaluate(self._valid_generator)
        self.save_score()

    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
        }
        save_json(
            path = Path("scores.json"),
            data=scores
        )

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
            })


            # --- NEW APPROACH FOR LOGGING MODEL ---
            # 1. Define a temporary path to save the model locally before logging
            # local_model_path = Path("local_model_to_log.keras")
            local_model_path = Path("VGG16Model.keras")
            # 2. Save the model directly using Keras's save method
            # This is where you specify the .keras extension explicitly
            # and avoid the deprecated 'save_format' argument.
            self.model.save(local_model_path)
            print(f"Model temporarily saved to: {local_model_path}")

            # 3. Log the saved model file as an artifact
            # You can still give it a name within MLflow artifacts (e.g., "model.keras")
            mlflow.log_artifact(str(local_model_path), artifact_path="model.keras")

            # 4. If you want to REGISTER the model, use mlflow.register_model
            # This points to the artifact you just logged.
            # Make sure you have the full artifact URI.
            # You might need to construct this based on your run_id
            # For simplicity, for now, we'll just log the artifact.
            # If you specifically need model registration, it's a bit more involved
            # to get the correct URI.
            # A common pattern is:
            logged_model_uri = f"runs:/{mlflow.active_run().info.run_id}/model.keras"
            mlflow.register_model(logged_model_uri, "VGG16Model")
            # --- END NEW APPROACH ---

            # Remove the previous mlflow.keras.log_model calls
            # if tracking_url_type_store != "file":
            #     mlflow.keras.log_model(
            #         self.model,
            #         "model.keras",
            #         registered_model_name="VGG16Model",
            #         keras_model_kwargs={"save_format": "keras"} # THIS IS NOW THE PROBLEM
            #     )
            # else:
            #     mlflow.keras.log_model(
            #         self.model,
            #         "model.keras",
            #         keras_model_kwargs={"save_format": "keras"} # THIS IS NOW THE PROBLEM
            #     )

            # Clean up the local temporary file
            if local_model_path.exists():
                os.remove(local_model_path)
                print(f"Cleaned up local model file: {local_model_path}")



            # if tracking_url_type_store != "file":
            #     mlflow.keras.log_model(
            #         self.model,
            #         "model.keras", # This is the artifact path within MLflow
            #         registered_model_name="VGG16Model",
            #         # ADD THIS ARGUMENT:
            #         keras_model_kwargs={"save_format": "keras"}
            #     )
            #     # mlflow.keras.log_model(self.model, "model.keras", registered_model_name="VGG16Model")

            # else:
            #     mlflow.keras.log_model(self.model, "model.keras"
            #         # self.model,
            #         # "model.keras"
            #     )
