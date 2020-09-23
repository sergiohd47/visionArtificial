
#PRIMERA PARTE DE LA PRACTICA (deteccion_orb.py)
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import glob
import sys

# FUNCION PROCESA LA IMAGEN PARA BUSCAR UN COCHE
def procesarImagen(imagen): 
    anchuraMatriz = imagen.shape[1]/10 #conseguir la altura-anchura de la imagen
    anchuraMatriz = math.trunc(anchuraMatriz) #redondeada la anchura-altura
    alturaMatriz = imagen.shape[0]/10 
    alturaMatriz = math.trunc(alturaMatriz) 
    matriz = np.zeros((anchuraMatriz, alturaMatriz)) #crea matriz acumuladora con 0
    keypoints, descriptores = orb.detectAndCompute(imagen, None) #sacan keypoints y descriptores de la imagen 
    for i in range(0, 100): #tenemos 100 descriptores y keypoints
        matches = flann.knnMatch(descriptores[i].reshape(1, 32), k = 6) #coge los k descriptores mas cercanos a ese descriptor
        keypointImagen = keypoints[i]
        for m in matches:
            for emparejamiento in m:
                keypointR = kpEntrenamiento[emparejamiento.imgIdx] #coges el keypoint del emparejamiento
                vectorR = vectorEntrenamiento[emparejamiento.imgIdx] #coges el vector del emparejamiento
                escalaTransformacion = keypointImagen.size/keypointR.size #saca la escala de transformacion vector
                vectorR[0] = escalaTransformacion*vectorR[0] #coordenada X-Y de la escala transformar el vector
                vectorR[1] = escalaTransformacion*vectorR[1] 
                anguloKeypointR = math.radians(keypointR.angle) #angulo del keypoint asociado al descriptor (radianes)
                anguloKeypointImagen = math.radians(keypointImagen.angle) 
                anguloVector = math.atan2(vectorR[1], vectorR[0]) #angulo del vector (radianes)
                anguloFinal = anguloVector-anguloKeypointR+anguloKeypointImagen
                moduloVector = math.sqrt(math.pow(vectorR[0], 2)+math.pow(vectorR[1], 2)) #modulo vector
                vectorR[0] = moduloVector*math.cos(anguloFinal) #coordenada X-Y vector con angulo del emparejamiento
                vectorR[1] = moduloVector*math.sin(anguloFinal) 
                keypointX = keypointImagen.pt[0]+vectorR[0] #coordenada x-Y del keypoint con el vector
                keypointY = keypointImagen.pt[1]+vectorR[1] 
                coordenadaX = math.trunc(keypointX/10) #redondea la coordenada X-Y
                coordenadaY = math.trunc(keypointY/10) 
                if ((coordenadaX<anchuraMatriz)&(coordenadaX>=0)&(coordenadaY<alturaMatriz)&(coordenadaY>=0)): #comparacion de las coordenadas estan en el rango dimensiones
                  matriz[coordenadaX][coordenadaY] = matriz[coordenadaX][coordenadaY]+1 #Aumenta en uno el valor de la posicion x e y del acumulador
    coordenadaX = np.argmax(np.max(matriz, axis = 1)) #cooordenada X-Y con el valor maximo de la matriz
    coordenadaY = np.argmax(np.max(matriz, axis = 0)) 
    coordenadaX = coordenadaX*10 #adapta coordenadas a la imagen
    coordenadaY = coordenadaY*10
    copiaImagen = imagen.copy()
    cv2.circle(copiaImagen, (coordenadaX, coordenadaY), 6, (255, 255, 255), -1, 16, 0) #dibuja  circulo+cuadrado en la imagen en coordenadas X-Y (https://docs.opencv.org/4.2.0/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9)
    cv2.rectangle(copiaImagen, (coordenadaX-20, coordenadaY-20), (coordenadaX+20, coordenadaY+20), (255, 255, 255), 6, 16, 0) 
    plt.imshow(cv2.cvtColor(copiaImagen,cv2.COLOR_BGR2RGB)) #parte para mostrar por pantalla (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.waitforbuttonpress.html#examples-using-matplotlib-pyplot-waitforbuttonpress)
    plt.waitforbuttonpress()

# FUNCION QUE PREPARA EL MATCHER
def prepararMatcher():
    listaKeypoints = [] #lista keypoints para el matcher
    listaVectoresVotacion = [] #lista vectores votacion para el matcher
    for imagen in listaImagenesEntrenamiento:
        listaKeypointsImagen, descriptores = orb.detectAndCompute(imagen, None) #se saca la lista de kp de la imagen y sus descriptores (enunciado)
        imagenX = (imagen.shape[1]/2) #coordenadas X-Y centro imagen
        imagenY = (imagen.shape[0]/2) 
        for keypoint in listaKeypointsImagen:
            listaKeypoints.append(keypoint)
            keypointX = keypoint.pt[0] #coordenadas X-Y del keypoint
            keypointY = keypoint.pt[1] 
            vectorX = imagenX-keypointX #vector keypoint relacion con el centro en eje X-Y
            vectorY = imagenY-keypointY 
            vectorCoordenadas = [] #coordenadas X-Y vector
            vectorCoordenadas.append(vectorX) 
            vectorCoordenadas.append(vectorY) 
            listaVectoresVotacion.append(vectorCoordenadas) #inserta vector coordenadas en una lista
        for descriptor in descriptores:
            flann.add([descriptor.reshape(1, 32)]) #se redimensiona el descriptor y se inserta en el FLANN
    flann.train() #inicializa el FLANN
    return listaKeypoints, listaVectoresVotacion

# PARTE PRINCIPAL DEL ARCHIVO
# se añade funcionalidad de meter argumentos por terminal pero se comenta porque no es lo pedido (http://www.holamundo.es/lenguaje/python/articulos/acceder-argumentos-pasados-parametro-python.html)
#if(len(sys.argv)==1):
  #parte donde se leen las imagenes de training
  #rutaEntrenamiento="/content/train" #ruta training
  #rutaEntrenamiento=sys.argv[1]
rutaEntrenamiento="train"
numeroImagenesEntrenamiento = len(glob.glob(rutaEntrenamiento+"/*.jpg")) #numero imagenes training (https://www.lawebdelprogramador.com/foros/Python/1557438-Contar-el-numero-de-elementos-de-una-carpeta-y-anadirlo-a-una-variable.html)
listaImagenesEntrenamiento = []
for i in range(1, numeroImagenesEntrenamiento+1):
  nombreImagen = rutaEntrenamiento+"/frontal_"+str(i)+".jpg"
  imagen = cv2.imread(nombreImagen, 0) #lee imagen en escala de grises
  if imagen is None: #comprobacion imagen es None
    print("Error cargando imagen training numero: "+i)
    break
  listaImagenesEntrenamiento.append(imagen) #inserta imagen leida en la lista de imagenes de entrenamiento
    
#crea el detector ORB con 100 keypoints (referencia enunciado)
orb = cv2.ORB_create(nfeatures = 100, scaleFactor = 1.3, nlevels = 4) 
#serie de parametros para crear el archivo FLANN
FLANN_INDEX_LSH = 6
parametrosIndex = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 3, multi_probe_level = 1)
parametrosBusqueda = dict(checks = -1)
flann = cv2.FlannBasedMatcher(parametrosIndex, parametrosBusqueda) #crea archivo FLANN con los parametros anteriores
#parte donde se hace el matcher
kpEntrenamiento, vectorEntrenamiento = prepararMatcher() 

#parte donde se leen las imagenes de testing
#rutaTest="/content/test" #ruta testing
#rutaTest=sys.argv[2]
rutaTest="test"
numeroImagenesTest = len(glob.glob(rutaTest+"/*.jpg")) #numero imagenes testing (https://www.lawebdelprogramador.com/foros/Python/1557438-Contar-el-numero-de-elementos-de-una-carpeta-y-anadirlo-a-una-variable.html)
listaImagenesTest = []
for i in range(1, numeroImagenesTest+1):
  nombreImagen = rutaTest+"/test"+str(i)+".jpg"
  imagen=  cv2.imread(nombreImagen,0) #lee la imagen en escala de grises
  #imagen1= ((imagen - imagen.min())/(imagen.max() - imagen.min()))*255 intento de aumentar el contraste lineal (NO FUNCIONA)
  if imagen is None: #comprobacion imagen es None
    print("Error cargando imagen testing numero: "+str(i))
    break
  listaImagenesTest.append(imagen) #Inserta la imagen en la lista de imagenes test

#parte donde ya se procesan todas las imagenes de coches
for i in range(0,numeroImagenesTest):
  procesarImagen(listaImagenesTest[i]) #procesa la imagen del coche test
#else:
#  print("Ayuda: python deteccion_orb.py rutaEntrenamiento rutaTest")