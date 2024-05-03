import os
import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from PIL import Image, ImageDraw

# Modelo
modelo = Sequential([
    Convolution2D(64, 5, 5, activation='relu', input_shape=(42,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(.20),
    Dense(4, activation='softmax')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

# Ruta al directorio que contiene los archivos JSON
directorio = "C:/Users/Carlos/AppData/Local/Google/Cloud SDK/"

# Obtener la lista de archivos en el directorio
archivos = os.listdir(directorio)

# Crear un diccionario para mapear las clases a índices numéricos
clases = {}
for idx, archivo in enumerate(archivos):
    if archivo.endswith(".ndjson"):
        clase = os.path.splitext(archivo)[0]
        clases[clase] = idx

# Obtener el número total de clases
num_clases = len(clases)

print("Número de clases: ", num_clases)

def procesar_archivo(archivo):
    with open(os.path.join(directorio, archivo), "r") as file:
        X_train = []
        y_train = []
        for line in file:
            drawing = json.loads(line)
            # Verificar si la palabra fue reconocida antes de procesar el dibujo
            if drawing['recognized']:
                # Crear una imagen en blanco
                img = Image.new('L', (256, 256), color='white')
                draw = ImageDraw.Draw(img)

                # Dibujar cada trazo en la imagen
                for stroke in drawing['drawing']:
                    # Modificar para manejar trazos con 2 o más puntos
                    for i in range(1, len(stroke[0])):
                        x0, y0 = stroke[0][i - 1], stroke[1][i - 1]
                        x1, y1 = stroke[0][i], stroke[1][i]
                        draw.line([(x0, y0), (x1, y1)], fill='black', width=5)

                # Redimensionar la imagen a 42x28 y convertirla a un array numpy
                img_resized = img.resize((28, 42))
                img_array = np.array(img_resized)[:, :, np.newaxis] / 255.0  # Normalizar los valores de píxel entre 0 y 1
                X_train.append(img_array)
                y_train.append(clases[drawing['word']])  # Suponiendo que tienes un diccionario que asigna un índice a cada palabra
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print("Datos creados")
        return X_train, y_train

# Entrenar el modelo con los datos procesados de los primeros tres archivos
for archivo in archivos[:3]:  # Procesar solo los primeros tres archivos
    if archivo.endswith(".ndjson"):
        X_train, y_train = procesar_archivo(archivo)
        print(f"Procesando y entrenando archivo: {archivo}")
        modelo.fit(X_train, y_train, batch_size=128, epochs=30, verbose=True)

# Guardar el modelo entrenado
modelo.save("modelo_entrenado.keras")

print("Entrenamiento finalizado. Modelo guardado como 'modelo_entrenado.keras'")

