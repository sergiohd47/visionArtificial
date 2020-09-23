#PRACTICA 2: LEER COCHE (RECORTAR MATRICULA+LEER DIGITOS+ESCRIBIR FICHERO)
import cv2
import glob
import os
import sys
import numpy as np
from operator import itemgetter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

H_MIN = 10 #Altura minima del contorno
CONST_UMBR = 30 #Pone la constante a 30

#FUNCION LEE IMAGENES TESTING
def importarImagenesTesting():
    ruta = sys.argv[1] + os.sep + "*.jpg" #ruta test (pasada por consola)
    listaImagenes = [] 
    for archivo in glob.glob(ruta): #recorre todos los archivos que se encuentran en ruta (https://docs.python.org/2/library/glob.html)
        nombre = os.path.basename(archivo) #separa la direccion: ruta y nombre del archivo
        imagen = cv2.imread(archivo, 0) #lee imagen gris
        imagenCompleta = [] #tupla de imagen y nombre
        imagenCompleta.append(imagen) 
        imagenCompleta.append(nombre) 
        listaImagenes.append(imagenCompleta) #inserta la tupla en la lista de imagenes
    return listaImagenes 

#FUNCION LEE IMAGENES TRAIN
def importarImagenesTraining():
    ruta = "." + os.sep + "training_ocr" + os.sep + "*.jpg" #ruta train (pasada por consola)
    listaImagenes = [] 
    for archivo in glob.glob(ruta): #recorre todos los archivos que se encuentran en ruta (https://docs.python.org/2/library/glob.html)
        nombre = os.path.basename(archivo) #separa la direccion: ruta y nombre del archivo
        imagen = cv2.imread(archivo, 0) #lee imagen gris
        imagenCompleta = [] #tupla imagen y nombre
        imagenCompleta.append(imagen)
        imagenCompleta.append(nombre) 
        listaImagenes.append(imagenCompleta) #inserta la tupla en la lista de imagenes
    return listaImagenes

#FUNCION FILTRA LOS CONTORNOS DE LA IMAGEN PARA ENCONTRAR LOS DE LA MATRICULA
def filtrarContornos(listaContornos, matricula):
    contornosMatricula = [] 
    contornosOrdenados = [] 
    for contorno in listaContornos: #recorre los contornos pasados
        coordenadaX, coordenadaY, coordenadaW, coordenadaH = cv2.boundingRect(contorno) #coordenadas contorno (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect)
        cnt = [] #tupla x y contorno
        cnt.append(coordenadaX) 
        cnt.append(contorno) 
        contornosOrdenados.append(cnt) #inserta la tupla en la lista de contornos
    contornosOrdenados = sorted(contornosOrdenados, key=itemgetter(0)) #ordenan los contornos en funcion de la x
    listaContornos = [fila[1] for fila in contornosOrdenados] #devuelve la posicion 1 de cada tupla de los contornos (devuelve los contornos)

    for contorno in listaContornos: 
        coordenadaX, coordenadaY, coordenadaW, coordenadaH = cv2.boundingRect(contorno) #coordenadas contorno (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect)
        coordenadaX1C = coordenadaX #x contorno
        coordenadaY1C = coordenadaY #y contorno
        coordenadaX2C = coordenadaX+coordenadaW #x+ancho contorno
        coordenadaY2C = coordenadaY+coordenadaH #y+alto contorno
        coordenadaX1M, coordenadaY1M, coordenadaX2M, coordenadaY2M = matricula #coordenadas de matricula
        if (coordenadaW < coordenadaH) & (coordenadaH >= H_MIN) & (coordenadaX1C >= coordenadaX1M) & (coordenadaX2C <= coordenadaX2M) & (coordenadaY1C >= coordenadaY1M) & (coordenadaY2C <= coordenadaY2M): #x,y estan dentro de la matricula y alto>ancho
            if (len(contornosMatricula) != 0):
                dentro = False #comprueba si hay contornos uno dentro de otro
                i = 0 
                for mContorno in contornosMatricula: 
                    coordenadaX1MC, coordenadaY1MC, coordenadaWMC, coordenadaHMC = cv2.boundingRect(mContorno) #coordenadas de mContorno (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect)
                    coordenadaX2MC = coordenadaX1MC+coordenadaWMC #x+ancho
                    coordenadaY2MC = coordenadaY1MC+coordenadaHMC #y+alto
                    if (coordenadaX1C >= coordenadaX1MC) & (coordenadaX2C <= coordenadaX2MC) & (coordenadaY1C >= coordenadaY1MC) & (coordenadaY2C <= coordenadaY2MC): # x,y estan dentro contorno
                        dentro = True 
                    elif (coordenadaX1C <= coordenadaX1MC) & (coordenadaX2C >= coordenadaX2MC) & (coordenadaY1C <= coordenadaY1MC) & (coordenadaY2C >= coordenadaY2MC): #x,y contienen contorno
                        contornosMatricula[i] = contorno #guarda nuevo contorno en posicion del antiguo
                        dentro = True
                    i = i+1 
                if (not dentro) & (len(contornosMatricula) <= 7): #si no esta dentro y len(contornos)<=7 (matriculas=longitud 7)
                    contornosMatricula.append(contorno)
            else: #si no hay contornos en contornosMatricula
                contornosMatricula.append(contorno) #Mete el contorno en el array
    contornosOrdenados = [] 
    for contorno in contornosMatricula:
        coordenadaX, coordenadaY,coordenadaW, coordenadaH = cv2.boundingRect(contorno) #coordenadas contorno (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect)
        cnt = [] #tupla x y contorno
        cnt.append(coordenadaX)
        cnt.append(contorno)
        contornosOrdenados.append(cnt) #inserta la tupla en la lista de contornos
    contornosOrdenados = sorted(contornosOrdenados, key=itemgetter(0)) #ordenan los contornos en funcion de la x
    contornosMatricula = [fila[1] for fila in contornosOrdenados] #devuelve la posicion 1 de cada tupla de los contornos (devuelve los contornos)
    return contornosMatricula #devuelve contornos de la matricula

#FUNCION CONSTRUIR VECTORES CARACTERISTICAS Y ETIQUETAS QUE USAREMOS PARA LDA
def construirVectoresImagenes(imagenesTrain):
    caracteristicas = [] 
    etiquetas = [] 
    for imagen in imagenesTrain: #recorre imagenes Train
        nombre = imagen[1] #nombre imagen
        etiqueta = nombre[:nombre.index('_')] #recorta nombre para conseguir etiqueta
        gray = imagen[0] #guarda imagen
        imagenUmbralizada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, CONST_UMBR) #umbralizado adaptativo gaussiano (https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold)
                                                                                                                                    #CONST_UMBR dependera de que carpeta se le pase (testing_ocr=30, testing_full_system=10)
        img = imagenUmbralizada
        contornos, jerarquia = cv2.findContours(imagenUmbralizada, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #contornos imagen umbralizada (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html)
                                                                                                         #devuelve tambien jerarquia pero nosotros no lo usamos
        if (contornos != []):
            cnt = contornos[0] #contorno posicion 0
            coordenadaX, coordenadaY, coordenadaW, coordenadaH = cv2.boundingRect(cnt) #coordenadas contorno
            img = img[coordenadaY:coordenadaY+coordenadaH, coordenadaX:coordenadaX+coordenadaW] #recorta el caracter
        imagenEscalada = cv2.resize(img, (10, 10), interpolation = cv2.INTER_LINEAR) #escala tamaño 10x10 (recomendacion enunciado)
        fila = np.reshape(imagenEscalada, (1, 100)) #caracter = fila 100 columnas
        caracteristica = fila[0] #coge la primera columna para sacar la caracteristica
        caracteristicas.append(caracteristica) 
        etiquetas.append(etiqueta) 
    return caracteristicas, etiquetas #devuelve las caracteristicas y las etiquetas


#PARTE PRINCIPAL DE LA PRACTICA  (main)
imagenesTest = importarImagenesTesting() #importa imagenes test
directorioRuta = sys.argv[1] #ruta imagenes para probar algoritmo
if ("testing_full_system" in sys.argv[1]): #si se utiliza testing_full_system en la llamada, se pone CONST_UMBR=10
    CONST_UMBR = 10 
directorioParametros = [] #parametros de ruta
directorioParametros = directorioRuta.split(os.sep) #separa en funcion de / (os.sep)
nombreArchivo = directorioParametros[-1] + ".txt" #nombre del archivo sera la ultima parte de la ruta
archivo = open(nombreArchivo, "w") #crea el fichero en modo escritura con el nombre de la ruta (https://programacion.net/articulo/escribir_y_leer_ficheros_en_python_1446)
imagenesTrain = importarImagenesTraining() #importa imagenes train
caracteristicas, etiquetas = construirVectoresImagenes(imagenesTrain) #construye vectores de caracteristicas y etiquetas
analizadorLDA = LinearDiscriminantAnalysis() #crea analizador LDA (linear discriminant analysis) (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
analizadorLDA.fit(caracteristicas, etiquetas) #entrena LDA con caracteristicas y etiquetas (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
clasificadorCoches = cv2.CascadeClassifier("haar" + os.sep + "coches.xml") #crea clasificador basandose en fichero .xml (enunciado)
for imagen in imagenesTest: 
    nombre = imagen[1] #nombre imagen
    imagenReal = imagen[0] #imagen

    cochesDetectados = clasificadorCoches.detectMultiScale(imagenReal, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)) #detecta coches independiente del tamaño      
    for (coordenadaX, coordenadaY, coordenadaW, coordenadaH) in cochesDetectados: 
    	cv2.rectangle(imagenReal, (coordenadaX, coordenadaY), (coordenadaX+coordenadaW, coordenadaY+coordenadaH), (255, 255, 255), 6) #pinta recuadro blanco en el morro
    	
   
    imagenUmbralizada = cv2.adaptiveThreshold(imagenReal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, CONST_UMBR) #umbralizado adaptativo gaussiano
    
    
    contornos, jerarquia = cv2.findContours(imagenUmbralizada, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #contornos imagen umbralizada (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html)
                                                                                                     #devuelve tambien jerarquia pero nosotros no lo usamos
    clasificadorMatriculas= cv2.CascadeClassifier("haar" + os.sep + "matriculas.xml") #crea clasificador matriculas con haar(haar: matriculas.xml)
    matriculas = clasificadorMatriculas.detectMultiScale(imagenReal) #detecta matriculas de la imagen (rectangulos que posiblemente sean matriculas)
    for i in range(0, len(matriculas)):
        coordenadaX, coordenadaY, coordenadaW, coordenadaH = matriculas[i] #coordenadas matricula
        cv2.rectangle(imagenReal, (coordenadaX, coordenadaY), (coordenadaX+coordenadaW, coordenadaY+coordenadaH ), (255, 255, 0), 3)
        matriculas[i][2] = coordenadaX+coordenadaW #coordenada extremo rectangulo eje x
        matriculas[i][3] = coordenadaY+coordenadaH #coordenada extremo rectangulo eje y
    if (len(matriculas) != 0):
        imagenCopia = imagenReal.copy()
        for matricula in matriculas:
            coordenadaXCentro = (matricula[0] + matricula[2]) / 2 #centro matricula eje x
            coordenadaYCentro = (matricula[1] + matricula[3]) / 2 #centro matricula eje y
            ancho = matricula[2] - matricula[0] #ancho matricula
            largoMatricula2 = ancho / 2 #largo matricula a la mitad
            contornosMatricula = filtrarContornos(contornos, matricula) #filtra contornos para tener contornos solo de la matricula
            if (len(contornosMatricula) != 0): #si encontramos contornos en la matricula
                resultadoMatricula = "" 
                contador = 0 
                for contorno in contornosMatricula: 
                    coordenadaX, coordenadaY, coordenadaW, coordenadaH = cv2.boundingRect(contorno) #coordenadas del contorno
                    digito = imagenUmbralizada[coordenadaY:coordenadaY+coordenadaH, coordenadaX:coordenadaX+coordenadaW] #digito de la imagen umbralizada
                    digito10x10 = cv2.resize(digito, (10, 10), interpolation = cv2.INTER_LINEAR) #escala el digito a 10x10 (recomendacion enunciado)
                    fila = np.reshape(digito10x10, (1, 100)) #digito=fila de 100 columnas
                    if (len(contornosMatricula) == 8) & (contador == 0): #si hay 8 contornos y estamos en el primer contorno, entonces ese contorno sera la E de ESPAÑA
                        resultado = "ESP"
                    else: 
                        aux = analizadorLDA.predict(fila) #predice la clase a la que pertenece ese digito
                        resultado = aux[0] #nombre de la clase digito
                        if (resultado == "O"): #Si detectamos una O como no puede haber vocales en Matriculas lo mas seguro es que sea un 0
                        	resultado = "0" #Asi que lo cambiamos por un 0 a la hora de escribir la matricula
                    resultadoMatricula += resultado 
                    contador += 1 
                    coordenadaX, coordenadaY, coordenadaW, coordenadaH = cv2.boundingRect(contorno) #coordenadas contorno
                    imagenCopia = cv2.rectangle(imagenCopia, (coordenadaX, coordenadaY), (coordenadaX+coordenadaW, coordenadaY+coordenadaH), (0, 255, 0), 2) #pinta contorno en la imagen
                    imagenCopia = cv2.circle(imagenCopia, (int(coordenadaXCentro), int(coordenadaYCentro)), 15, (255, 255, 0), 1) 
                    cv2.imshow("Nombre: "+nombre, imagenCopia) 
                cv2.waitKey(0) #espera pulsacion tecla
                cv2.destroyAllWindows()
            archivo.write(nombre + " ") #escribe nombre en el fichero (https://programacion.net/articulo/escribir_y_leer_ficheros_en_python_1446)
            archivo.write(str(coordenadaXCentro) + " ") #escribe centro(coordenada x) de la matricula en el fichero (https://programacion.net/articulo/escribir_y_leer_ficheros_en_python_1446)
            archivo.write(str(coordenadaYCentro) + " ") #escribe centro(coordenada y) de la matricula en el fichero (https://programacion.net/articulo/escribir_y_leer_ficheros_en_python_1446)
            archivo.write(resultadoMatricula + " ") #escribe resultado de la matricula en el fichero (https://programacion.net/articulo/escribir_y_leer_ficheros_en_python_1446)
            archivo.write(str(largoMatricula2) + "\n") #escribe largo matricula a la mitad en el fichero (https://programacion.net/articulo/escribir_y_leer_ficheros_en_python_1446)
archivo.close()

