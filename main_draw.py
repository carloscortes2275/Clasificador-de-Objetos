import tkinter as tk
from tkinter import colorchooser
from PIL import Image
from PIL import ImageGrab
from tkinter import filedialog
import datetime
import os
#IA
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cv2

# Variables globales
last_x, last_y = None, None
color = "black"
shape = "pencil"
ObjectClass = ""
brush_size = 2
eraser_size = 10
existing_lines = []
background1 = "lightblue"
background2 = "white"
background3 = "lightyellow"
    
def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    if shape == "pencil":
        line = canvas.create_line((last_x, last_y, event.x, event.y), fill=color, width=brush_size, capstyle=tk.ROUND, smooth=tk.TRUE)
    elif shape == "eraser":
        line = canvas.create_line((last_x, last_y, event.x, event.y), fill="white", width=eraser_size, capstyle=tk.ROUND, smooth=tk.TRUE)
    existing_lines.append(line)
    last_x, last_y = event.x, event.y

def set_tool(tool):
    global shape
    shape = tool
    # Deshabilitar todos los botones de herramientas
    pencil_button.config(state="normal")
    eraser_button.config(state="normal")
    # Habilitar el botón de la herramienta seleccionada
    if tool == "pencil":
        pencil_button.config(state="disabled")
    elif tool == "eraser":
        eraser_button.config(state="disabled")

def choose_color():
    global color
    new_color = colorchooser.askcolor()
    if new_color[1]:
        color = new_color[1]
        color_button.config(bg=color)  # Actualizar color del botón

def update_brush_size():
    global brush_size
    try:
        new_size = int(brush_size_entry.get())
        if new_size < 1:
            brush_size = 2
            brush_size_entry.delete(0, tk.END)
            brush_size_entry.insert(0, "2")
        else:
            brush_size = new_size
    except ValueError:
        brush_size = 2
        brush_size_entry.delete(0, tk.END)
        brush_size_entry.insert(0, "2")

def update_eraser_size():
    global brush_size
    try:
        new_size = int(eraser_size_entry.get())
        if new_size < 1:
            brush_size = 2
            eraser_size_entry.delete(0, tk.END)
            eraser_size_entry.insert(0, "10")  # Restaurar valor predeterminado
        else:
            brush_size = new_size
    except ValueError:
        brush_size = 10
        eraser_size_entry.delete(0, tk.END)
        eraser_size_entry.insert(0, "10")

def reset_draw():
    canvas.delete(tk.ALL)
    ObjectClass = ""
    label.config(text="La clase de la imagen es: " + ObjectClass)
    root.update()

def predict():
    now = datetime.datetime.now()
    filename = f"imagen_{now.strftime('%Y%m%d_%H%M%S')}.png"
    directory = "D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/loads/generated_images"
    # Combinar el nombre del archivo con el directorio
    filepath = os.path.join(directory, filename)
    # Guardar la imagen
    x = root.winfo_rootx() + canvas.winfo_x() + 250
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + 1100
    y1 = y + 1000


    screenshot = ImageGrab.grab(bbox=(x, y, x1, y1))
    screenshot.save(filepath, format="PNG")
    # Diccionario que mapea los números de clase a las etiquetas de clase
    class_labels = {0: 'basketball', 1: 'pencil', 2: 'sofa'}
    # Cargar el modelo
    modelo = load_model("D:/Documentos/Clasificacion Inteligente de Datos/Proyecto Reconocimiento de Objetos/modelo/modelo.keras")
    print('Modelo cargado')
    # Cargar la imagen de entrada
    img_path = filepath
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

    # Actualizar el texto del label ObjectClass
    ObjectClass = predicted_label
    label.config(text="La clase de la imagen es: " + ObjectClass)
    root.update()

root = tk.Tk()
root.title("Dibujar")
root['bg'] = 'lightgrey'
root.attributes('-transparentcolor', 'grey')

###################################################

# Frame izquierdo Herramientas de dibujo
left_frame = tk.Frame(root, bg=background1, width=50, height=400)
left_frame.pack(side="left", fill="y")  # No expande

# Herramientas de dibujo en el frame izquierdo
label = tk.Label(left_frame, text="Herramientas de dibujo\n", justify="left", font=("Arial", 14), bg=background1)
label.pack()

# Frame para el botón Pencil y el entry de tamaño
pencil_frame = tk.Frame(left_frame, bg=background1)
pencil_frame.pack()

# Botón para seleccionar el lápiz
pencil_button = tk.Button(pencil_frame, text="Lapiz", command=lambda: set_tool("pencil"), state="disabled")
pencil_button.pack(side="left", padx=5, pady=2)  # Espacio horizontal de 5 píxeles

# Campo de entrada para el tamaño del pincel
brush_size_entry = tk.Entry(pencil_frame, width=2)
brush_size_entry.insert(0, "2")
brush_size_entry.pack(side="left")

label = tk.Label(pencil_frame, text="px", bg=background1)
label.pack()

# Frame para el botón Eraser y el entry de tamaño
eraser_frame = tk.Frame(left_frame, bg=background1)
eraser_frame.pack()

# Botón para seleccionar la goma de borrar
eraser_button = tk.Button(eraser_frame, text="Borrador", command=lambda: set_tool("eraser"))
eraser_button.pack(side="left", padx=5, pady=2) # Espacio horizontal de 5 píxeles

# Campo de entrada para el tamaño de la goma de borrar
eraser_size_entry = tk.Entry(eraser_frame, width=2)
eraser_size_entry.insert(0, "10")  # Valor predeterminado
eraser_size_entry.pack(side="left")

label = tk.Label(eraser_frame, text="px", bg=background1)
label.pack()

# Frame para el label color y el color
color_frame = tk.Frame(left_frame, bg=background1)
color_frame.pack()

label = tk.Label(color_frame, text="Color: ", bg=background1)
label.pack(side="left", padx=5)

# Botón para seleccionar el color
color_button = tk.Button(color_frame, text="", bg=color, width=4, height=1, command=choose_color)
color_button.pack(side="left", padx=2, pady=5)

#########################################################

# Frame central para el lienzo de dibujo
canvas_frame = tk.Frame(root, bg=background1)
canvas_frame.pack(side="left", fill="both", expand=True)

# Lienzo de dibujo en el frame central
canvas = tk.Canvas(canvas_frame, bg=background2, width=600, height=800)
canvas.pack(fill="both", expand=False)

#########################################################

# Frame derecho Predicción
right_frame = tk.Frame(root, bg=background1, width=200, height=800)
right_frame.pack(side="left", fill="both", expand=True)

label = tk.Label(right_frame, text="Predicción de la clase", justify="left", font=("Arial", 14), bg=background1)
label.pack()

# Frame para predecir y reset
pre_res_frame = tk.Frame(right_frame, bg=background1)
pre_res_frame.pack()

save_button = tk.Button(pre_res_frame, text="Resetear Dibujo", command=reset_draw)
save_button.pack(side="left", padx=5, pady=10)

reset_button = tk.Button(pre_res_frame, text="Predecir", command=predict)
reset_button.pack(side="left", padx=5)

# Frame para la etiqueta y cuadro clase
save_frame = tk.Frame(right_frame, bg=background1)
save_frame.pack()

label = tk.Label(save_frame, text="Clase: " + ObjectClass, wraplength=200, justify="left", font=("Arial", 12), bg=background1)
label.pack(side="left", padx=5, pady=20)

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)

brush_size_entry.bind("<Return>", lambda event: update_brush_size())  # Actualizar tamaño del pincel al presionar Enter en el campo de entrada
eraser_size_entry.bind("<Return>", lambda event: update_eraser_size())

root.mainloop()