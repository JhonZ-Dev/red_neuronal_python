import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Cargar y preparar datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar píxeles a valores entre 0 y 1
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # Convertir etiquetas a one-hot encoding
