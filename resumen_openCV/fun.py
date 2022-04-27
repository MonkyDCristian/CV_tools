import cv2
import numpy as np

# call
img = cv2.imread("img_name")

# cambiar tipo de imagen
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pasar filtro
img_blur = cv2.GaussianBlur(img_gray,(7,7),0)
# más en: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

# gradiente: img, min grad, max grad
img_canny = cv2.canny(img, 150, 200)

# dilatacion y erocion
kernel = np.ones((5,5), np.unit8)
img_dilation = cv2.dilate(img_canny, kernel, interations = 1)
img_eroded = cv2.erode(img_canny, kernel, interations = 1)
# más en: https://unipython.com/transformaciones-morfologicas/

# resize img
img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
# más en: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

#invertir imagen: 0 = vertical, 1 = horizontal, -1 = diagonal
orientation = 1 #
img = cv2.flip(img, orientation)
# más en https://www.geeksforgeeks.org/python-opencv-cv2-flip-method/

# obtener partes de la imagen
img_part = img[0:img.shape[0]//2, 0:img.shape[1]//2]

# dibujar figuras y texto
# https://unipython.com/funciones-dibujar-opencv/

# transformaciones geometricas
# https://unipython.com/transformaciones-geometricas-de-imagenes-con-opencv/

# funciones de segmentación (Thresholding)
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

# transformada de fourier
# https://unipython.com/transformada-de-fourier/

# join img
# horizontal
2_img_h = np.hstack(img,img)
# vertical
2_img_v = np.vstack(img,img)

# matrices de 1 y 0
matriz_0 = np.zeros((5,5), dtype = int )
matriz_1 = np.ones((5,5), dtype = int)

# matris con dimenciones de una imagen
matris = np.zeros_like(img) # or np.ones_like(img)