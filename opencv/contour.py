import cv2 as cv
import numpy as np
img = cv.imread('photo/Sample.jpg')

cv.imshow("imagen", img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

#blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
#cv.imshow("cosa", blur)
#
#canny = cv.Canny(blur, 125, 155)
#cv.imshow("CANNY", canny)


ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

cv.imshow("thresh", thresh)
countours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE  )

cv.drawContours(blank, countours, -1, (0,0,255), 1 )
cv.imshow('draw', blank)

print(len(countours))
cv.waitKey(0)