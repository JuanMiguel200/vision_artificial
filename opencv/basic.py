import cv2 as cv

img = cv.imread("photo/jojo's.jpg")

cv.imshow("cdd", img)

#grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("pp", gray)


#blur
blur = cv.GaussianBlur(img,(3,3), cv.BORDER_DEFAULT)
cv.imshow("pp", blur)

cv.waitKey(0)