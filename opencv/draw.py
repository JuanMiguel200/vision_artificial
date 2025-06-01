import cv2 as cv
import numpy as np

blank = np.zeros((720,1080, 3), dtype='uint8')
cv.imshow('jojo', blank)  

#paint the image

#blank[200:300,300:400] = 0,255,0

cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0), thickness=-1)

cv.imshow('jojo', blank) 
cv.waitKey(0) 