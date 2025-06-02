import cv2 as cv
 
#img = cv.imread("photo/jojo's.jpg")

#cv.imshow('karakusu', img)

capture = cv.VideoCapture(0)

while(True):
    isTrue, frame = capture.read()
    cv.imshow('VIdeo', frame)
    if cv.waitKey(20) and 0xFF==('d'):
        break



cv.destroyAllWindows()
cv.waitKey(0)