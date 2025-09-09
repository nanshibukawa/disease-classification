import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):

        # self.model = tf.keras.applications.vgg16.VGG16(
        # self.model = tf.keras.applications.vgg19.VGG19(
        #     input_shape=self.config.params_image_size,
        #     weights=self.config.params_weights,
        #     include_top=self.config.params_include_top,
        # )
        # self.save_model(path=self.config.base_model_path, model=self.model) 


        if self.config.params_model_name == "VGG16":
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top,
            )
        elif self.config.params_model_name == "VGG19":
            self.model = tf.keras.applications.vgg19.VGG19(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top,
            )
        elif self.config.params_model_name == "MobileNetV3Large":
            self.model = tf.keras.applications.MobileNetV3Large(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top,
            )
        else:
            raise ValueError(f"Unsupported model name: {self.config.params_model_name}")
        
        self.save_model(path=self.config.base_model_path, model=self.model) 


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Forçar o formato .keras ao salvar o modelo
        if not str(path).endswith('.keras'):
            path = Path(str(path).replace('.h5', '.keras'))
        model.save(path)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, dropout_rate=0.5):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Arquitetura condicional baseada no dropout_rate
        if dropout_rate == 0.0:
            # Arquitetura simples (original) quando dropout é 0
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation='softmax'
            )(flatten_in)
        else:
            # Arquitetura com dropout: adicionar dropout ANTES da camada de saída
            # Manter estrutura similar à original, apenas adicionando dropout
            
            # Camada de dropout aplicada diretamente no flatten
            dropout_layer = tf.keras.layers.Dropout(
                rate=dropout_rate,
                name='dropout_before_output'
            )(flatten_in)
            
            # Camada de saída final (igual à versão sem dropout)
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation='softmax'
            )(dropout_layer)

        full_model = tf.keras.Model(
            inputs=model.input, 
            outputs=prediction)
        
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
            metrics=['accuracy']
        )
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        # Usar dropout_rate do config se disponível, senão usar 0.5 como padrão
        dropout_rate = getattr(self.config, 'params_dropout_rate', 0.5)
        
        self.full_model = self._prepare_full_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till= None,
            learning_rate = self.config.params_learning_rate,
            dropout_rate = dropout_rate
        )
        print("Summary of the model about to be saved:")
        self.full_model.summary()  # Adicionado para inspecionar o modelo antes de salvar
        # Inspecionar todas as camadas do modelo antes de salvar
        for layer in self.full_model.layers:
            print(f"Layer: {layer.name}, Type: {type(layer)}, Config: {layer.get_config()}")
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

