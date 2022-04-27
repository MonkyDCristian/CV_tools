import cv2
import numpy as np

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])

    for y in range(rows):
        for x in range(cols):
            if len(imgArray[y][x].shape) == 2:
                imgArray[y][x] = cv2.cvtColor(imgArray[y][x], cv2.COLOR_GRAY2BGR)

            if x == 0:
                img_hor = cv2.resize(imgArray[y][x], scale, interpolation=cv2.INTER_AREA)
            else:
                img_hor = np.hstack((img_hor, cv2.resize(imgArray[y][x], scale, interpolation=cv2.INTER_AREA)))

        if y == 0:
            img_ver = img_hor
        else:
            img_ver = np.vstack((img_ver, img_hor))

    return img_ver
