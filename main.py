import cv2 as cv
import numpy as np
import random

imagen_original = cv.imread("exam2.png", cv.IMREAD_GRAYSCALE)

# Deteccion de bordes
imagen_suavizada = cv.blur(imagen_original, (5,5))
_, imagen_binarizada = cv.threshold(imagen_suavizada, 150, 255, cv.THRESH_BINARY_INV)

# Deteccion de objetos convexos
contornos,_ = cv.findContours(imagen_binarizada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contorno = sorted(contornos, key=cv.contourArea, reverse=True)[0]
resultado = np.zeros((imagen_original.shape[0], imagen_original.shape[1], 3), dtype=np.uint8)

poi = cv.convexHull(contorno)

# Area de interes
color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
cv.drawContours(resultado, [poi], -1, color, 2)

# Segmentacion del punto de interes
mascara = np.zeros((imagen_binarizada.shape[0] , imagen_binarizada.shape[1]), dtype=np.uint8)
cv.fillPoly(mascara, [poi], 255)
region = cv.bitwise_and(imagen_suavizada, imagen_suavizada, mask=mascara)
x, y, w, h = cv.boundingRect(poi)
recorte_final = region[y:y+h, x:x+w]

# Correccion de perspectiva con tecnica homografica
def ordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype="float32")
    puntos = puntos.reshape(4, 2)
    
    suma = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(suma)]
    rect[2] = puntos[np.argmax(suma)]
    
    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]
    rect[3] = puntos[np.argmax(diff)]
    
    return rect

perimetro = cv.arcLength(poi, True)
epsilon = 0.02 * perimetro
contorno_examen = cv.approxPolyDP(poi, epsilon, True)

# si no da 4 puntos, ir subiendo epsilon hasta lograrlo
intentos = 0
while len(contorno_examen) != 4 and intentos < 10:
    epsilon *= 1.1
    contorno_examen = cv.approxPolyDP(poi, epsilon, True)
    intentos += 1

if len(contorno_examen) != 4:
    raise ValueError("No se pudo aproximar el contorno a 4 puntos")

rectangulo = ordenar_puntos(contorno_examen)
(tl, tr, br, bl) = rectangulo

anchuraA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
anchuraB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
max_ancho = max(int(anchuraA), int(anchuraB))

alturaA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
alturaB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
max_alto = max(int(alturaA), int(alturaB))

destino = np.array([
    [0, 0],
    [max_ancho - 1, 0],
    [max_ancho - 1, max_alto - 1],
    [0, max_alto - 1]
], dtype="float32")

matriz = cv.getPerspectiveTransform(rectangulo, destino)
examen_aplanado = cv.warpPerspective(imagen_suavizada, matriz, (max_ancho, max_alto))

cv.imshow("Imagen",imagen_binarizada)
cv.imshow("Imagen2",recorte_final)
cv.imshow("Examen Aplanado", examen_aplanado)

cv.waitKey(0)
cv.destroyAllWindows()