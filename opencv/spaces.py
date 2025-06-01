import cv2 as cv
img = cv.imread('photo/Sample.jpg')

cv.imshow("imagen", img)

#BGR TO GRAYSCALE

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow("hsv", hsv)

#BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

cv.imshow("hsv", lab)

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

cv.imshow("hsv", rgb)

cv.waitKey(0)