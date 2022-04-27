import cv2

cap = cv2.VideoCapture(0) #"video name
cv2.namedWindow('video') # window name

while True:
    succ, img = cap.read()
    if succ:
        cv2.imshow('video',img)

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
