
#SEGUNDA PARTE DE LA PRACTICA (deteccion_haar.py)
import matplotlib.pyplot as plt
import cv2
import sys
import glob

#FUNCION PROCESA LA IMAGEN PARA BUSCAR UN COCHE
def procesarImagen(imagen, rutaClasificador):
  clasificadorCoches = cv2.CascadeClassifier(rutaClasificador) #crea clasificador basandose en fichero .xml (enunciado)
  cochesDetectados = clasificadorCoches.detectMultiScale(imagen, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)) #detecta coches independiente del tamaño
  for (coordenadaX, coordenadaY, coordenadaW, coordenadaH) in cochesDetectados: 
    cv2.rectangle(imagen, (coordenadaX, coordenadaY), (coordenadaX+coordenadaW, coordenadaY+coordenadaH), (255, 255, 255), 6) #pinta recuadro blanco en el morro del coche (https://docs.opencv.org/4.2.0/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9)
  plt.imshow(cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB))#parte para mostrar por pantalla (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.waitforbuttonpress.html#examples-using-matplotlib-pyplot-waitforbuttonpress)
  plt.waitforbuttonpress()

#PARTE PRINCIPAL DEL ARCHIVO
#se añade funcionalidad de meter argumentos por terminal pero se comenta porque no es lo pedido (http://www.holamundo.es/lenguaje/python/articulos/acceder-argumentos-pasados-parametro-python.html)
#if(len(sys.argv)==3):
  #parte donde se leen las imagenes de training
  #rutaTest="/content/test" #ruta testing
  #rutaClasificador=sys.argv[1]
rutaClasificador="haar/coches.xml"
  #rutaTest=sys.argv[2]
rutaTest="test"
numeroImagenesTest = len(glob.glob(rutaTest+"/*.jpg")) #numero imagenes training (https://www.lawebdelprogramador.com/foros/Python/1557438-Contar-el-numero-de-elementos-de-una-carpeta-y-anadirlo-a-una-variable.html)
listaImagenesTest = []
for i in range(1, numeroImagenesTest+1):
  nombreImagen = rutaTest+"/test"+str(i)+".jpg"
  imagen=  cv2.imread(nombreImagen,0) #lee la imagen en escala de grises
  if imagen is None: #comprobacion imagen es None      
    print("Error cargando imagen testing numero: "+str(i))
    break
  listaImagenesTest.append(imagen) 
#parte donde ya se procesan todas las imagenes de coches
for i in range(0,numeroImagenesTest):
  procesarImagen(listaImagenesTest[i],rutaClasificador) #procesa la imagen del coche test
#else:
#  print("Ayuda: python deteccion_haar.py rutaClasificador rutaTest")