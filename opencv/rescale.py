import cv2 as cv
img = cv.imread('photo/chihina.jpg')
cv.imshow('La mala para chii', img)

def rescale (frame, scale = 2.0):
    width = int(frame.shape[1] * scale)
    heigh = int(frame.shape[0] * scale)
    dimensions = (width,heigh)
    return cv . resize(frame,dimensions, interpolation=cv.INTER_AREA)
resized_img = rescale(img)
cv.imshow('img', resized_img )
cv.waitKey(0)
def changeres(width, height):
    capture.set(3,width)
    capture.set(4,height)
    


capture = cv.VideoCapture('video/hyakkano-100-girlfriends.gif')

while(True):
    isTrue, frame = capture.read()
    frame_resixed = rescale(frame)
    
    cv.imshow('VIdeo', frame)
    cv. imshow('resixed', frame_resixed)
    if cv.waitKey(20) and 0xFF==('d'):
        break
    
capture.release()
cv.destroyAllWindows()
