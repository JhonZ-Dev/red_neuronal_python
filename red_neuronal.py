import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Cargar y preparar datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar píxeles a valores entre 0 y 1
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # Convertir etiquetas a one-hot encoding
# Construir el modelo de red neuronal
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))  # Capa de aplanamiento para convertir imágenes 28x28 en un vector de 1D
model.add(layers.Dense(128, activation='relu'))  # Capa densa con 128 unidades y función de activación ReLU
model.add(layers.Dropout(0.2))  # Capa de dropout para regularización
model.add(layers.Dense(10, activation='softmax'))  # Capa de salida con 10 unidades para las 10 clases y función de activación softmax
# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nPrecisión en el conjunto de prueba: {test_acc}')

# Hacer predicciones en algunas imágenes de prueba
predictions = model.predict(x_test[:5])
print('\nPredicciones:')
for i, prediction in enumerate(predictions):
    predicted_label = tf.argmax(prediction).numpy()
    true_label = tf.argmax(y_test[i]).numpy()
    print(f'Imagen {i + 1}: Predicción={predicted_label}, Etiqueta verdadera={true_label}')