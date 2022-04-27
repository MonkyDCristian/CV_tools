#https: // www.youtube.com/watch?v = WQeoO7MI0Bs & list = PLMoSUbG1Q_r9p7iYBg6z6tZP002DAJ41H & index = 1

import cv2
import time
import numpy as np
from stack_img import stackImages

def get_contours(img, img_cont):
    nothing = True
    # detecta los puntos que forman una figura de una imagen canny
    # contours: conjunto de puntos de cada fig
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # calcula en area de cada figura
        area = cv2.contourArea(cnt)
        #print(area)
        # si el area es mayor a 500 pix
        if area > 500:
            #  dibuja el contorno de la figura
            #cv2.drawContours(img_cont, cnt, -1,(255,0,0), 3)
            
            # calcula el perimetro de la figura cerrrada
            peri = cv2.arcLength(cnt,True)
            # retorna los puntos que conforman la figura
            approx = cv2.approxPolyDP(cnt, 0.03*peri, True)
            
            objlen = len(approx)
            # retona las esquina inferior de donde parte la figura junto
            # con su ancho y largo
            x,y,w,h = cv2.boundingRect(approx)

            # si el objeto tiene tres esquinas entonces es un triagulo
            if objlen == 3: objtype = "tri"
            # si tiene 4 entonces es un cuadrado o un rectangulo 
            elif objlen == 4:
                prop = w/float(h)
                if 0.95< prop < 1.05: objtype = "square"
                else: objtype = "rec"
            # en cualquier otro caso es un circulo 
            else:  objtype = "cir"
            
            cv2.drawContours(img_cont, approx, -1, (0, 0, 255), 20)
            #cv2.rectangle(img_cont,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.putText(img_cont, objtype, (x + 10,y + 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 2)
            
            nothing = False
    
    if nothing:
        return np.array([0])
    
    else:
        return approx

def pre_processing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7,7), 1)
    img_canny = cv2.Canny(img_blur, 100, 100)
    #kernel = np.ones((3,3))
    #img_dil = cv2.dilate(img_canny, kernel, iterations=1)
    
    return img_gray, img_canny

def reorder(points):
    
    rezide_points = points.reshape((4,2))
    
    add = rezide_points.sum(1)
    diff = np.diff(rezide_points, axis=1)
    
    pos = [np.argmin(add), np.argmin(diff), np.argmax(diff), np.argmax(add)]
    
    return points[pos]
    
def perspective_transform(img, x, y, points):
    
    points = reorder(points)
    
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (x, y))
    
    return dst

def get_object(img):
    img_cont = img.copy()

    img_gray, img_pre = pre_processing(img)

    approx = get_contours(img_pre, img_cont)
    
    print(approx)
    
    if len(approx) != 1:
        dst = perspective_transform(img, 100, 100, approx)

        #print(img.shape, img_pre.shape, img_pre,  img_cont.shape) # [[img, img_gray, img_pre, img_cont]])
        full = stackImages((500, 380), [[img_cont, dst]])
    
    else: 
        full = stackImages((500, 380), [[img_cont]])
    
    return full
    
def main(path="log3.png"):
    
    pTime = time.time()
    cTime = 0
    
    while True:
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        img = cv2.imread(path)
        full = get_object(img)
        
        #cv2.putText(full, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.imshow('video', full)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

def main_loop(camera=0):
    pTime = time.time()
    cTime = 0
    
    cap = cv2.VideoCapture(camera)  # "video name
    cv2.namedWindow('video')  # window name

    while True:
        succ, img = cap.read()
        if succ:
            full = get_object(img)
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            #cv2.putText(full, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            
            cv2.imshow('video', full)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
        
main()
