import tensorflow as tf
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

num_classes = 8

# Rutas a los directorios de train, val y test
train_dir = 'D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/dataset/train'
val_dir = 'D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/dataset/val'
test_dir = 'D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/dataset/test'

# Crear generadores de datos de imágenes
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse')

# Crear modelo
modelo = tf.keras.models.Sequential([
    #Flatten(input_shape=(28, 28, 3)),
    #tf.keras.layers.Dense(units=128, activation='relu'),
    #tf.keras.layers.Dense(units=10, activation='softmax')
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes es el número de clases en tus datos
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

# Resumen del modelo
modelo.summary()

# Entrenar el modelo
modelo.fit(train_generator, epochs=20, validation_data=val_generator)

# Evaluar el modelo
test_loss, test_accuracy = modelo.evaluate(test_generator)

print("Exactitud en el conjunto de prueba:", test_accuracy)

# Guardar el modelo
modelo.save("D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/modelo/modelo.keras")
print('Modelo guardado')

# Obtener el mapeo de clases a índices numéricos del generador de datos
class_indices = train_generator.class_indices

# Invertir el mapeo para obtener un diccionario de índices numéricos a clases
num_to_class = {v: k for k, v in class_indices.items()}

# Imprimir el mapeo de índices numéricos a clases
print("Mapeo de índices numéricos a clases:", num_to_class)
