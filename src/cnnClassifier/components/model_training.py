import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import numpy as np

from cnnClassifier.entity.config_entity import TrainingConfig
from sklearn.metrics import f1_score



class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.best_f1 = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_gen = self.validation_data
        y_true = val_gen.classes
        y_pred_probs = self.model.predict(val_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f'val_f1_score: {f1:.4f}')
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()
        logs = logs or {}
        logs['val_f1_score'] = f1

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        # Recompile o modelo com um novo otimizador após carregar
        self.model.compile(
            # optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config.params_learning_rate, weight_decay=1e-4),

            # loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
            loss=tf.keras.losses.CategoricalCrossentropy(),

            metrics=['accuracy']
        )
    
    def train_valid_generator(self):
        """
        Cria geradores para treino e validação usando dados já divididos manualmente
        """
        # Verificar se existe divisão manual dos dados
        split_train_dir = Path("artifacts/data_split/train")
        split_val_dir = Path("artifacts/data_split/validation")
        
        if split_train_dir.exists() and split_val_dir.exists():
            # Usar divisão manual (sem validation_split)
            datagenerator_kwargs = dict(rescale=1./255)
        else:
            # Fallback para divisão automática
            datagenerator_kwargs = dict(
                rescale=1./255,
                validation_split=0.2
            )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Configurar gerador de validação
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        if split_val_dir.exists():
            # Usar dados de validação separados manualmente
            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=split_val_dir,
                shuffle=False,
                **dataflow_kwargs
            )
        else:
            # Usar subset validation (divisão automática)
            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )

        # Configurar gerador de treinamento
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                brightness_range=[0.2,1.0],
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        if split_train_dir.exists():
            # Usar dados de treino separados manualmente
            self.train_generator = train_datagenerator.flow_from_directory(
                directory=split_train_dir,
                shuffle=True,
                **dataflow_kwargs
            )
        else:
            # Usar subset training (divisão automática)  
            self.train_generator = train_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="training",
                shuffle=True,
                **dataflow_kwargs
            )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        f1_callback = F1ScoreCallback(self.valid_generator)

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[f1_callback]

        )



        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )