'''--------------------------
# Face detection : License belongs to Intel  #
---------------------------'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator

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

    '''offset = w/20
    eye_y = np.int32(y + h*0.3)
    eye_h = np.int32(h*0.15)
    left_eye_x = np.int32(x + w/5 - offset)
    right_eye_x = left_eye_x + np.int32(w*2/5)
    eye_w = np.int32(w/5 + 2*offset)
    return [[left_eye_x,right_eye_x,eye_y,eye_w, eye_h]]'''

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
thresh_high = 120
thresh_low = 0.4*thresh_high
'''cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)'''
'''frameCount = 0'''
eh=np.int32(0)
ew=np.int32(0)
skippedFrame = 0

# Optimization - First face detection we use Haar cascade face detector and subsequent face detection
# we pply Kalman filter for speed concern.

while (cap.isOpened()):
    ret, img = cap.read()
    if ret: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''frameCount = frameCount + 1'''
        '''print 'frame # ', frameCount'''
        #gray = cv2.GaussianBlur(gray, (15,15), 0)  #Filter

        faces = face_cascade.detectMultiScale(gray, 1.3,5) #Return multiple faces if there are. 

        if len(faces): 
            # Pending Only when faces are detected, then proceed eye search, otherwise skip this iteration.
            index, value = max(enumerate(faces[:,2]),key=operator.itemgetter(1)) #Find max according to height w.
            faces = faces[index,:]
            x = faces[0]
            y = faces[1]
            w = faces[2]
            h = faces[3]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            eyes = eye_region(x,y,w,h)
            for (lex, rex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (lex,ey), (lex+ew, ey+eh), (0,255,0),2)
                cv2.rectangle(img, (rex,ey), (rex+ew, ey+eh), (0,255,0),2)
                left_eye_roi = gray[ey:ey+eh, lex:lex+ew]
                right_eye_roi = gray[ey:ey+eh, rex:rex+ew]
                left_eye_color = img[ey:ey+eh, lex:lex+ew]
                right_eye_color = img[ey:ey+eh, rex:rex+ew]
        
            '''Hough circle detection'''
            
            lcircles = cv2.HoughCircles(left_eye_roi,cv2.HOUGH_GRADIENT,1,10,
                        param1=65,param2=22,minRadius=5,maxRadius=20)         
            rcircles = cv2.HoughCircles(right_eye_roi,cv2.HOUGH_GRADIENT,1,10,
                        param1=65,param2=22,minRadius=5,maxRadius=20)
           
            #If there are circles found, then we proceed          
            if lcircles is not None and rcircles is not None:
                lcircles = np.uint16(np.around(lcircles))
                '''left eye'''
                for i in lcircles[0,:]:
                    # draw the outer circle
                    cv2.circle(left_eye_color,(i[0],i[1]),i[2],(0,255,255),2)
                    # draw the center of the circle
                    cv2.circle(left_eye_color,(i[0],i[1]),2,(0,0,255),2)

                '''right eye'''
                rcircles = np.uint16(np.around(rcircles))
                for i in rcircles[0,:]:
                    # draw the outer circle
                    cv2.circle(right_eye_color,(i[0],i[1]),i[2],(0,255,255),2)
                    # draw the center of the circle
                    cv2.circle(right_eye_color,(i[0],i[1]),2,(0,0,255),2)

            else:
                '''No circles found, pupil diameter is zero'''
                skippedFrame += 1
                print skippedFrame

        else:
            left_eye_color=np.zeros([eh, ew, 3])
            right_eye_color=np.zeros([eh, ew, 3])

       
        '''Below this line, we can write more to get required variables'''
        cv2.imshow('img',img)
        cv2.imshow('left eye edges', left_eye_color)
        cv2.imshow('right eye edges', right_eye_color)
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        
    else:
        break 


cap.release()
cv2.destroyAllWindows()



