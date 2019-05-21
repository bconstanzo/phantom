# Detector de Objetos
*dlib* es una toolkit para lograr aprendizaje automatizado y en este script se lo usará para crear un __detector de objetos semirrígidos basado en HOG__. Este objeto está implementado como una máquina de soporte vectorial o SVM.

La idea es poder ver la facilidad con la que se puede generar una SVM pensada para resolver este tipo de problemas, analizando los resultados para dos casos particulares: un objeto con muchas características como lo es una taza, y la cara de osos polares.

#### HOG, SVM y procesamiento de imágenes.
El histograma de gradientes orientados o HOG es un descriptor de características que se queda con la información más *útil* de una imagen de tamaño ancho x alto x 3 (canales) y la convierte en un vector de características de largo n. Este vector se usa como entrada para entrenar una SVM, en este caso el detector de objetos. El modelo va a aprender de los datos ingresados (imágenes y rectángulos de detección), para luego poder hacer uso de la generalización y determinar salidas para nuevas imágenes.

#### Etapas.

![alt text](https://i.imgur.com/RXXPudP.png "Entrenamiento")

*dlib* nos provee de ciertos objetos y funciones que nos permite crear un detector de manera muy sencilla. La función que vamos a utilizar para entrenar el detector es:

```python
dlib.train_simple_object_detector(dataset_filename, detector_output_filename, options)
```
* `dataset_filename`: path del archivo .xml que contiene información de las distintas imágenes junto a las coordenadas de los rectángulos. Existen varias herramientas que proveen cierta facilidad para generar este archivo. Una buena herramienta es [imgLab](http://imglab.ml).
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
___
#### Pruebas realizadas.
##### Tazas.
El caso de las tazas es un problema de detección un poco difícil, debido a que una taza puede tener muchas formas y colores. Se hicieron dos pruebas dentro de este caso, una entrenando al modelo solo con tazas vistas desde frente (quedando la manija a la izquierda - derecha) y otra prueba con imágenes de tazas desde otras perspectivas. Estas imágenes de entrenamiento se encuentran en [tazas](/tools/object%20detector/tazas) junto a los archivos .xml correspondientes.

###### Prueba 1
Se utilizaron 4 imágenes para el entrenamiento con las características mencionadas. No hubo buenos resultados, esto se puede deber a los pocos datos ingresados para el entrenamiento y también por el problema de las tazas en si.

![](https://github.com/bconstanzo/phantom/blob/master/tools/object%20detector/tazas/prueba1/Foto%2017-5-19%2013%2018%2058.jpg)
###### HOG obtenido:
![](https://github.com/bconstanzo/phantom/blob/master/tools/object%20detector/tazas/prueba1/hog.png "HOG tazas - pruebas")

###### Prueba 2
Se basó en el uso de dos imágenes más de entrenamiento para el modelo (6), donde estas fotos se tomaron con otras perspectivas. Alimentar al modelo con estas nuevas imágenes no dieron buenos resultados tampoco. Sigue sin poder detectar los mismos casos que para que la primer prueba.

![](https://github.com/bconstanzo/phantom/blob/master/tools/object%20detector/tazas/prueba2/Foto%2017-5-19%2013%2018%2049.jpg?raw=true "Ejemplo2")

Para ambas pruebas el conjunto de datos de test se compuso de 8 imágenes (13 tazas en total) inéditas para el modelo, de las cuales se obtuvieron los siguientes resultados:

| Prueba              | Falsos positivos           | Falsos negativos  |
| -------------       |:--------------------------:|:-----------------:|
| 4 imágenes          | 2                          | 10                |
| 6 imágenes (persp)  | 3                          |   10               |

Ambos modelos detectaron 3/13 tazas = 23% de los objetos.

##### Osos polares.
Para esta prueba se proponen 14 imágenes de osos polares donde se pueden ver las caras de frente de los mismos. Solo se consideraron para los rectángulos de detección las caras de osos polares adultos. Las imágenes de entrenamiento junto con el train.xml se encuentran en el repositorio en [osos_polares](/tools/object%20detector/osos_polares)

###### HOG obtenido:
![](https://github.com/bconstanzo/phantom/blob/master/tools/object%20detector/osos_polares/hog.png "HOG osos polares")

El conjunto de datos de test que se utilizó se compone de 18 imágenes, las cuales contienen 19 caras de osos polares *adultos*.

![](https://github.com/bconstanzo/phantom/blob/master/tools/object%20detector/osos_polares/result.png "Ejemplo detección oso polar")

| Prueba                | Falsos positivos           | Falsos negativos  |
| -------------         |:--------------------------:|:-----------------:|
| Osos polares          | 1                          | 4                |

El modelo pudo detectar 13/19 = 70% de las caras de osos. Un resultado muy bueno comparándolo con el caso de las tazas. Hay que tener en cuenta que se usó una cantidad mayor de imágenes para el entrenamiento.

Lo bueno de `simple_object_detector` de *dlib* es que con relativamente pocas imágenes de entrenamiento se obtienen buenos resultados.

___
### Referencias
[1] https://www.learnopencv.com/histogram-of-oriented-gradients/

[2] https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/

[3] http://blog.dlib.net/2014/02/dlib-186-released-make-your-own-object.html

[4] http://dlib.net/python/index.html#dlib.train_simple_object_detector

