# Detector de Objetos
*dlib* es una toolkit para lograr aprendizaje automatizado y en este script se lo usará para crear un __detector de objetos semirrígidos basado en HOG__. Este objeto está implementado como una máquina de soporte vectorial o SVM.

La idea es poder ver la facilidad con la que se puede generar una SVM pensada para resolver este tipo de problemas, analizando los resultados para dos casos particulares: un objeto con muchas características como lo es una taza, y la cara de osos polares.

#### HOG, SVM y procesamiento de imagenes.
El histograma de gradientes orientados o HOG es un descriptor de características que se queda con la informacion más *util* de una imagen de tamaño ancho x alto x 3 (canales) y la convierte en un vector de características de largo n. Este vector se usa como entrada para entrenar una SVM, en este caso el detector de objetos. El modelo va a aprender de los datos ingresados (imagenes y rectangulos de deteccion), para luego poder hacer uso de la generalización y determinar salidas para nuevas imagenes.

### Etapas.

![alt text](https://i.imgur.com/RXXPudP.png "Entrenamiento")

*dlib* nos provee de ciertos objetos y funciones que nos permite crear un detector de manera muy sencilla. La función que vamos a utilizar para entrenar el detector es:

```python
dlib.train_simple_object_detector(dataset_filename, detector_output_filename, options)
```
* `dataset_filename`: path del archivo .xml que contiene informacion de las distintas imágenes junto a las coordenadas de los rectángulos. Existen varias herramientas que proveen cierta facilidad para generar este archivo. Una buena herramienta es [imgLab](http://imglab.ml).
* `detector_output_filename`: el detector de objetos entrenado es serializado en un archivo .svm.
* `options`: *dlib* proporciona un objeto contenedor de las opciones para la rutina de training, estas mismas vienen con valores por defecto razonables. Algunas de estas opciones son:

```python
options.C = 5
options.epsilon = 0.01
options.add_left_right_image_flips = True
```

![alt text](https://i.imgur.com/GUN6rXu.png "Consulta")

Ya tenemos el detector de objetos entrenado. Ahora deberíamos tener un conjunto de imágenes listas para con el detector. *dlib* proporciona herramientas para cargar imágenes y mostrarlas por pantallas, también para a partir de ciertos coordenadas (nuestra salida del detector) dibujar sobre una imagen un rectángulo. 

La interacción con el detector es muy sencilla, simplemente se le pasa la imagen y devuelve las coordenadas del objeto(s).
```python
detector = dlib.simple_object_detector("detector.svm")
salida = detector(imagen)
```
#### Pruebas realizadas.
##### Tazas.
El caso de las tazas es un problema de detección un poco difícil, debido a que una taza puede tener muchas formas y colores. Se hicieron dos pruebas dentro de este caso, una entrenando al modelo solo con tazas vistas desde frente (quedando la manija a la izquierda - derecha) y otra prueba con imagenes de tazas desde otras perspectivas. Estas imagenes de entrenamiento se encuentran en [tazas](/tazas)
junto a los archivos .xml correspondientes.

###### Prueba 1
Se utilizaron 4 imagenes para el entrenamiento con las caracteristicas mencionadas. No hubo buenos resultados, esto se puede deber a los pocos datos ingresados para el entrenamiento y también por el problema de las tazas en si.

![alt text](/tazas/prueba1/Foto 17-5-19 13 18 58.jpg "Ejemplo1")
![alt text](/tazas/prueba1/hog.png "HOG tazas - pruebas")

###### Prueba 2
Se basó en el uso de dos imágenes más de entrenamiento para el modelo (6), donde estas fotos se tomaron con otras perspectivas. Alimentar al modelo con estas nuevas imágenes no dieron buenos resultados tampoco. Sigue sin poder detectar los mismos casos que para que la primer prueba.

![alt text](/tazas/prueba2/Foto 17-5-19 13 18 49.jpg "Ejemplo2")

Para ambas pruebas el conjunto de datos de test se compuso de 8 imagenes (13 tazas en total) ineditas para el modelo, de las cuales se obtuvieron los siguientes resulados:

| Prueba              | Falsos positivos           | Falsos negativos  |
| -------------       |:--------------------------:|:-----------------:|
| 4 imágenes          | 2                          | 10                |
| 6 imágenes (persp)  | 3                          |   10               |

Ambos modelos detectaron 3/13 tazas = 23% de los objetos.

##### Osos polares.
Para esta prueba se proponen 14 imágenes de osos polares donde se pueden ver las caras de frente de los mismos. Solo se consideraron para los rectangulos de detección las caras de osos polares adultos. Las imagenes de entrenamiento junto con el train.xml se encuentran en el repositorio en [oso_polares](/osos_polares)

![alt text](/osos_polares/hog.png "HOG osos polares")

El conjunto de datos de test que se utilizó se compone de 18 imágenes, las cuales contienen 19 caras de osos polares *adultos*.
![alt text](/osos_polares/result.png "Ejemplo detección oso polar")

| Prueba                | Falsos positivos           | Falsos negativos  |
| -------------         |:--------------------------:|:-----------------:|
| 15 imágenes           | 1                          | 4                |

El modelo pudo detectar 13/19 = 70% de las caras de osos.

Como __conclusión__ se puede obtener que en este caso el detector de objetos está funcionando bien. Lo bueno del simple_object_detector de dlib es que con relativamente pocas imágenes de entrenamiento se obtienen buenos resultados.

___
### Referencias
[1] https://www.learnopencv.com/histogram-of-oriented-gradients/
