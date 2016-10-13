'''--------------------------
# Face detection : License belongs to Intel  #
---------------------------'''
import numpy as np
import cv2

"""
Geometric method to locate eye region.
vertically  : middle 1/4 to 1/2 of face is eye region
horizontally: 1/6 to 3/6 left eye, 3/6 to 5/6 right eye
"""
def eye_region(x,y,w,h):
    offset = w/20
    eye_y = np.int32(y + h/4)
    eye_h = np.int32(h/4)
    left_eye_x = np.int32(x + w/5 - offset)
    right_eye_x = left_eye_x + np.int32(w*2/5)
    eye_w = np.int32(w/5 + 2*offset)
    return [[left_eye_x,right_eye_x,eye_y,eye_w, eye_h]]

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


''' multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades'''
''' https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml'''
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
''' https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml'''

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
cap = cv2.VideoCapture(0)
'''cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)'''
#frameCount = 0

# Optimization - First face detection we use Haar cascade face detector and subsequent face detection
# we pply Kalman filter for speed concern.

while 1:
    tic()
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #frameCount = frameCount + 1
    #print 'frame # ', frameCount
    #gray = cv2.GaussianBlur(gray, (15,15), 0)  #Filter 
    faces = face_cascade.detectMultiScale(gray, 1.3,5) #Return multiple faces if there are.
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      
        eyes = eye_region(x,y,w,h)
        for (lex, rex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (lex,ey), (lex+ew, ey+eh), (0,255,0),2)
            cv2.rectangle(img, (rex,ey), (rex+ew, ey+eh), (0,255,0),2)
        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    toc()

cap.release()
cv2.destroyAllWindows()



