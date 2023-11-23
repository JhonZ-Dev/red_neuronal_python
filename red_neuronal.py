import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Cargar y preparar datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
