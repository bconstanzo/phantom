# Uso de descriptores para encontrar objetos en imágenes

La idea de este trabajo fue desarrollar un programa que permita detectar objetos
en imágenes a partir del uso de descriptores previstos por 
[OpenCV](https://opencv.org/ "OpenCV") (en el caso de las prubas, se lo utilizó 
para detectar logos). Estos pueden estar una o varias veces, completos o de 
manera parcial por obstrucciones, al mismo tiempo que rotados y/o en diferentes
escalas.

-----
## ¿Qué descriptor fue el más conveniente?
Para este trabajo se consideró el uso de distintos detectores y descriptores de
imágenes, optando en primer lugar por los de uso libre, como por ejemplo
__BRISK__, __ORB__, __FREAK__ y __AKAZE__, pero ninguno porporcionó los
resultados necesitados, por lo que fueron descartados. Finalmente, luego de
todas las pruebas realizadas, se decidió utilizar __SIFT__ como el descriptor de
este trabajo, ya que a pesar de ser más lento que algunos de los mencionados
previamente y tener la gran contra de ser patentado, fue el que entregó mejores
resultados.

A raíz de esto, habría que hacer la aclaración de que para este trabajo, se
partió de un [tutorial de OpenCV](https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html "Feature Matching + Homography to find Objects")
acerca de __SIFT__, donde se explica su uso básico.

-----
### Descripción general del algoritmo original
En el tutorial mencionado, se muestra el código para detección de objetos en
imágenes, utilizando como ejemplo la portada de un libro. En primer lugar se
cargan las imágenes (en este ejemplo el libro y el escenario donde se 
encuentra), se crea un objeto [__SIFT__](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform "Scale invatiant feature transform"),
y este objeto se encarga de describir ambas imágenes calculando sus puntos clave
(keypoints) y una serie de descriptores para cada uno de estos puntos, que luego
son utilizados para la comparación y reconocimiento.

Una vez que está hecha esta descripción, se utiliza el
[__FlannBasedMatcher__](https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html "FlannBasedMatcher Class Reference")
para detectar las relaciones entre los descriptores de puntos de cada imagen. Si
estas relaciones son lo suficientemente buenas, se consideraran un match, siendo
que lo descripto en la primer imagen *puede ser* lo mismo que lo descripto en la
segunda imagen. Este es un paso crucial, ya que en caso de tener un mal matcheo,
esto puede llevar a que haya una coincidencia con el logo cuando no está, o
también, que a pesar de que el logo este presente en la imagen, no se lo
encuentre.

Para esto, el __FlannBasedMatcher__ utiliza un algoritmo basado en kdtree, y
entrega un listado de matches (coincidencias entre descriptores) ordenados por
distancia (el primer match es siempre el mas cercano). A su vez, el parametro
`k` determina la cantidad de matches deseados por cada punto de la primer
imagen, siendo `k=2` en el código original (para cada descriptor de la primer
imagen, se encuentran los dos descriptores mas cercanos en la segunda).

Luego de esto, se filtran los matches que se consideran correctos. Para esto, se
compara la distancia del primer match con la del segundo. Esto es para detectar
las características únicas dentro de la imagen, y evitar el posible matcheo con
areas parecidas. Si el primer match tiene una distancia lo suficientemente menor
que el segundo (en este caso se calcula si es menor al 70% del segundo), se lo
considera "unico" y se lo mantiene. En caso contrario, se lo descarta.

Finalmente, luego de aplicar este filtrado, si la cantidad de matches
resultantes es mayor que un número preestablecido (`MIN_MATCH_COUNT = 10` en el
código) se considera que se encontró el logo. Después de esto se calcula la
transformación para marcar el logo detectado, y se dibuja una serie de lineas
que unan los puntos clave de las dos imágenes.

-----
### Problemas con el algoritmo
Luego de ver esto y probarlo con una serie de imágenes, nos dimos cuenta de tres
problemas principales frente a lo que necesitábamos solucionar:

1) El sistema no soporta __múltiples apariciones__ de un objeto en la misma
   imagen. En caso de que se encuentre varias veces en la imagen, el algoritmo
   lo considera como una sola, y es probable que la transformación para marcado
   falle, además de no poder informar la cantidad de apariciones.
2) El matcheo es imperfecto, por lo que existe la posibilidad de que se
   encuentre como match un punto incorrecto, lejano del resto de los matches
   correctos, y este deforme la transformación, o en casos peores, hayan
   suficientes __matches incorrectos__ repartidos a lo largo de la imagen como
   para considerar que el logo esta presente, cuando no está.
3) Existe la posibilidad de que varios puntos de la primer imagen (el logo en
   nuestro caso), sean __matcheados con el mismo punto__ de la segunda imagen,
   llegando a ser 10 o mas, por lo que se considera que el logo esta presente, a
   pesar de que es en un solo pixel.

Además de esto, vale la pena mencionar que el tamaño de la imagen original
impacta de manera directa en su detección. Mientras mayor sea la resolución de
la misma, más puntos y descriptores van a ser generados, por lo que es mas fácil
encontrarla de manera parcial. Esto lleva a que la cantidad de falsos positivos
aumente en gran medida. Mientras mas pequeña sea, mas específica sera la
búsqueda.

Para llegar a esta conclusión, se partio de la busqueda de la foto de un libro
con un tamaño relativamente grande para la búsqueda (1280x960), y se fue
reduciendo su tamaño en un 20%. Mientras esto pasaba, se buscaba su coincidencia
en 5 imágenes, donde el libro solo se encontraba realmente en una de ellas
(*completa.jpeg*), en otras dos habían libros presentes con caracteristicas
similares (*box.png* y *box_in_scene.png*), y en las últimas dos no había ningun
tipo de forma relacionada a un libro (*auto.jpg* y *pasto.jpg*) *(Todas las
imagenes se encuentran en la carpeta de imagenes de prueba)*. En la siguiente
tabla se muestra, para cada búsqueda, la resolución de la imagen y la imagen en
la que fue buscada, devolviendo la cantidad de matches encontrados en cada caso.

| libro.jpeg       | (1280x960) | (1024x768) | (768x576) | (512x384) | (256x192) |
| ---------------- |:----------:|:----------:|:---------:|:---------:|:---------:|
| completa.jpeg    |    136     |     94     |    65     |    50     |    15     |
| auto.jpg         |    248     |    119     |     9     |     6     |     0     |
| box.png          |    140     |    119     |    35     |    19     |     1     |
| box_in_scene.png |    161     |     72     |    28     |    10     |     0     |
| pasto.jpg        |     50     |     27     |     4     |     1     |     0     |

Como se ve claramente, en el primer caso pareciera que el libro es encontrado en
todas las imágenes de prueba, e incluso se podría decir que, según la cantidad
de matches encontrados, es más probable que esté presente en la imágenes
erroneas en lugar de la correcta. Esto cambia drasticamente a medida que se
reduce la resolución de la foto, llegando a detectarlo únicamente en la imagen
correcta. Esto no quiere decir que mientras más pequeña la imagen mejor, sino
que el tamaño correcto va a depender de los casos a analizar, porque también hay
que considerar que la cantidad de matches en la imagen correcta fue reduciendo.

-----
### Soluciones planteadas
- #### Multiples apariciones
    El código, como ya dicho previamente, solo se encarga de detectar si el logo
    esta presente en la imagen e intenta marcarlo, pero no detecta la cantidad
    de veces que esta presente, ni puede marcar el logo varias veces. Para
    solucionar esto, se plantearon tres alternativas:
    1) #### Mas matches por punto
        En primer lugar, lo que se planteó fue detectar múltiples matches por
        punto de la primer imagen. Como se dijo previamente, el
        __FlannBasedMatcher__ tiene un parametro llamado `k` que determina la
        cantidad de matches por punto. Como alternativa, se estableció el valor
        de `k=10`, pensando en detectar el logo con un máximo de 9 veces.
        Tambien se hicieron modificaciones en el código siguiente, para que en
        el filtrado no se considere solo la distancia del primer match contra la
        del segundo, sino que se consideren todos los matches obtenidos hasta
        encontrar una diferencia del 70% con el match siguiente (si nunca se
        encuentra esa diferencia, se descartan todos los matches de ese punto).
        Esta fue la implementación de ese filtrado:

        ```python
        matches = flann.knnMatch(desc1,desc2,k=2)

        good_matches = []
        for match in matches:
            for i, (m, n) in enumerate(zip(match, match[1:]), start=1):
                if m.distance < 0.7*n.distance:
                    break
            if match[i-1].distance < 0.7*match[i].distance:
                good_matches.extend(match[:i])
        ```

        Esto llevo a dos problemas principales, que derivaron en el descarte de
        esta alternativa. Primero, aumentaba la posibilidad de que varios puntos
        de las imágenes estuvieran matcheados con un mismo punto de la otra.
        Esto significa, por ejemplo, que en casi todos los casos, varios puntos
        de la primer imagen estaban matcheados con el mismo punto de la segunda
        imagen, y que varios puntos de la segunda imagen podían estar matcheando
        incorrectamente con el mismo punto de la primera. A su vez, también
        aumentó considerablemente la cantidad de matcheo incorrecto.
        Consecuencia de todo esto, el algoritmo encontraba el logo en
        prácticamente todas las imágenes. 


    2) #### Clustering
        ##### Por matches:
        Descartado esto, se consideró la alternativa de realizar clustering,
        tanto de matches como de keypoints en base a las distancias entre sí.
        Esto, para generar distintos grupos de matches, y asi considerar la
        posible aparición del logo multiples veces en la misma imagen. En el
        caso del clustering de matches, nunca se llegó a implementar porque se
        consideró traía el mismo problema que el caso anterior. Se seguía
        necesitando aumentar el valor de `k` en el matcher, con lo que mantenía
        todos los problemas ya mencionados. Si no se hiciera esto, se seguiría
        encontrando solo un match por punto de la primer imagen, con lo que la
        posible detección múltiple se vería dificultada.

        ##### Por keyPoints:
        Finalmente, se implementó el clustering de keypoints. Esto significa
        que, luego de describir las imágenes con __SIFT__, se utilizó la
        herramienta de __DBSCAN__ para generar estos clusters. Así, se generaron
        los grupos en base a las posiciones de los Keypoints, y una distancia
        máxima que los agregaba a cada cluster. Como parámetros se consideraron
        un Epsilon en base al tamaño del eje menor de la imagen, multiplicado
        por un valor `e` (más adelante se habla de este valor), y que en cada
        cluster debían existir al menos 3 puntos (son necesarios al menos 3
        puntos para la transformación y marcado del logo en un plano). Todos los
        puntos que quedaran fuera de clusters de al menos 3 elementos, son
        descartados.

        Luego de esto, se generan los matches por cluster, con lo que, en cada
        cluster, se puede detectar una aparición del logo, manteniendo el
        parametro `k=2` y evitando así los problemas del caso anterior. El nuevo
        problema que se presenta en esta alternativa es que el cluster no sea ni
        demasiado grande y ocupe toda la imagen, lo que haría regresar al
        problema original donde solo se podría encontrar una aparición del logo,
        ni que el cluster sea demasiado pequeño, lo que posibilitaría que se
        encuentre el logo varias veces de manera erronea, debido a que varios
        clusters se encuentren en diferentes partes de la misma aparición del
        logo (un cluster para cada letra del logo, por ejemplo). Para esto,
        luego de algunas pruebas, se estableció como valor por defecto
        `e=0.025`, que tiene una baja taza de fallos de detección múltiple, y
        evita generar un cluster demasiado grande en la mayoría de las imágenes.
        De todos modos, este valor es configurable, y depende principalmente del
        tamaño y tipo de imágenes.

- #### Descarte de matches concentrados en un punto, y de matches lejanos
    Luego de solucionar el problema anterior, se utilizó nuevamente clustering
    con __DBSCAN__ para eliminar matches lejanos al resto, considerando que
    estos sólo son uno aislado y probablemente erróneo, que generaría problemas
    al calcular la transformación. Para esto, se sigue considerando que el
    cluster debe mantener al menos tres matches (sino sería descartado), y en el
    caso del Epsilon, se opto por que este fuera la distancia promedio entre
    todos los puntos, con lo que solo se descartarían aquellos que se encuentren
    demasiado alejados del resto. Al mismo tiempo, si luego del cálculo, ee
    Epsilon resultara igual a 0, esto significaría que todos los matches del
    cluster estarían en el mismo punto, por lo que sería descartado (solo en el
    caso de que TODOS esten en el mismo punto). También, si se diera la
    posibilidad y los matches estuvieran agrupados en distancias suficientes, se
    generarían nuevos clusters para agruparlos.

    Después de todo esto, sigue existiendo la posibilidad de que en el cluster
    hayan varios matches con el mismo punto de la segunda imagen, y es imposible
    saber cual de ellos es el correcto y cual erróneo, por lo que no se puede
    descartar ninguno. A momento de determinar si el objeto está presente en la
    imagen, por ejemplo, cuando hay 8 matches en el mismo punto y 3 más
    repartidos en áreas cercanas, originalmente, el algorítmo habría aceptado
    como aparición al haber 11 matches en total. Por esta razón se optó por
    contar sólo los matches en distintos puntos, considerando que de los 8 que
    se encuentran en el mismo lugar, solo uno de ellos podría ser correcto, y
    los otros 7 no. Finalmente, en este caso de ejemplo, el recuento sería de 4,
    por lo que se considera que el logo no está presente.

-----
### Acerca de su uso
Después de la descripción de como funciona, en esta sección se hará mención a
como se utiliza este código, junto con el significado de algunos de los
parámetros principales. Hay dos funciones que se encargan de todo el proceso,
invocando al resto de las funciones cuando es necesario: `find_object_in_image`
y `find_object_in_directory`.

Ambas reciben como primer parámetro una imagen que se desea encontrar (en los
casos de prueba, el logo). No es recomendable que esta imagen sea demasiado
grande, ya que al describirla, se encuentran demasiados puntos clave y
descriptores. Esto lleva a que el proceso sea mas lento, además de aumentar la
cantidad de falsos positivos porque se encuentran pequeñas partes de esta imagen
mas facilmente, sin la necesidad de encontrala de manera global. Mientras mas
expecífica, mejor.

En segundo lugar, `find_object_in_image` recibe otra imagen que debe escanear, y
en cámbio `find_object_in_directory` recibe una ruta a un directorio para
procesar todas las imagenes que se encuentren en él. Además de esto, ambas
pueden recibir como parámetro el valor del Epsilon para el clustering de
Keypoints (e=Valor) y se puede dibujar los circulos en las posiciones de los
matches al pasar como verdadero el parametro drawKM (drawKM=True).

La función `find_object_in_image` devuelve el número de apariciones del objeto
en la segunda imagen, y su modificación con el objeto marcado en ella en caso de
haber sido encontrado.

La función `find_object_in_directory` es un generator, por lo que devuelve los
resultados a medida que se lo necesita. Estos son los mismo valores, la cantidad
de veces que se encontro el objeto en la imagen, y su modificación.

-----
### Referencias
[1] OpenCV - https://opencv.org/

[2] Introduction to SIFT (Scale-Invariant Feature Transform) - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html

[3] FlannBasedMatcher Class Reference - https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html

[4] Feature Matching + Homography to find Objects - https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

[5] SIFT (Scale-invariant feature transform) - https://en.wikipedia.org/wiki/Scale-invariant_feature_transform

[6] Distinctive Image Features from Scale-Invariant Keypoints - https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf