import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cv2

# Diccionario que mapea los números de clase a las etiquetas de clase
class_labels = {0: 'bed', 1: 'chair', 2: 'sofa', 3: 'swivelchair', 4: 'table'}

# Cargar el modelo
modelo = load_model("D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/modelo/modelo.keras")
print('Modelo cargado')

# Cargar la imagen de entrada
img_path = 'D:\Documentos\Clasificacion Inteligente de Datos\Proyecto Reconocimiento de Objetos\pruebas\imagenPrueba3.jpg'
test_img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)

## Preprocesar la imagen
test_img1 = cv2.resize(test_img1,(28,28))
test_input1 = test_img1.reshape((1,28,28,3))

# Realizar la predicción
predictions = modelo.predict(test_input1)

# Obtener la clase predicha
predicted_class = np.argmax(predictions[0])

# Obtener la etiqueta de clase predicha
predicted_label = class_labels[predicted_class]

print("La clase predicha es:", predicted_label)


'''import tensorflow as tf
import numpy as np
from PIL import Image

# Código para cargar el modelo y hacer predicciones con nuevas imágenes
modelo_cargado = tf.keras.models.load_model("D:/Documentos/Clasificación Inteligente de Datos/Proyecto Reconocimiento de Objetos/modelo/modelo.keras")

#Predicción de imagen nueva
new_image_path = "D:/Documentos/Clasificación Inteligente de Datos/Proyecto Reconocimiento de Objetos/pruebas/imagenPrueba1.png"

# Cargar la imagen usando PIL
imagen_pil = Image.open(new_image_path)

# Convertir la imagen a escala de grises
imagen_pil_grayscale = imagen_pil.convert('L')

# Verificar las dimensiones de la imagen cargada
#print("Dimensiones de la imagen cargada:", imagen_pil_grayscale.size)

# Redimensionar la imagen a 28x28 píxeles
imagen_np = np.array(imagen_pil_grayscale.resize((28, 28))) / 255.0

# Agregar una dimensión de lote para que coincida con la forma esperada por el modelo
imagen_np = np.expand_dims(imagen_np, axis=0)

# Realiza la predicción con el modelo cargado
prediccion = modelo_cargado.predict(imagen_np)

# La predicción es un array de probabilidades para cada clase (0-9)
# Encuentra la clase con la probabilidad más alta
#clase_predicha = np.argmax(prediccion)

print("La clase predicha es:", prediccion)'''