import cv2
import numpy as np
from stack_img import stackImages

def empty(value):
    print(value)

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 600, 240)

cv2.createTrackbar("Hue min","TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue max","TrackBars", 179, 179, empty)
cv2.createTrackbar("sat min","TrackBars", 0, 255, empty)
cv2.createTrackbar("sat max","TrackBars", 255, 255, empty)
cv2.createTrackbar("val min","TrackBars", 0, 255, empty)
cv2.createTrackbar("val max","TrackBars", 255, 255, empty)

# color detection
while True:
    img = cv2.imread("img/log3.png")
    print(img)
    img_copy = img.copy()
    
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue max","TrackBars")
    s_min = cv2.getTrackbarPos("sat min","TrackBars")
    s_max = cv2.getTrackbarPos("sat max","TrackBars")
    v_min = cv2.getTrackbarPos("val min","TrackBars")
    v_max = cv2.getTrackbarPos("val max","TrackBars")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imghsv, lower, upper)
    canny = cv2.Canny(mask, 50, 50)
    dil = cv2.dilate(canny, np.ones((5,5)), iterations=1)
    
    img_copy[np.where(dil > 100)] = np.array([0, 0, 255])
    
    full = stackImages((500, 380), [[img, imghsv, mask, img_copy]])
    cv2.imshow("full", full)
    cv2.waitKey(1)

# 26 - 180 G
