'''
Software name : pupil_track_contour_v1.py
Developed by: Amanda Ye on 18-Feb-2017
current version finds pupil boundary contour

19-Feb-2017
> Print computational time for each iteration of pupil contour detection.
    Max 96ms @frame124.
> Fit ellipse to contour.
> Save timestamp and pupil major axis (in pixels) to txt file.
        

'''


### Contour method to detect pupil
## Pipeline
## 1. grayscale
##
## 2. median blur (may not needed)
##
## 3. canny edge and sobel direction of gradients
##
## 4. Find contours
## 4.1 Discard contour with large arc perimeters.
##
## 5. Gradient entropy, select maximum entropy among all contour candidates
## 5.1. plot contour (if jittered, try to momentum reduce jitterness)
##
## 6. Fit ellipse.
##
## 7. Save major axis (in pixels) and timestamp to txt file.

'''--------------------------
# Face detection : License belongs to Intel  #
---------------------------'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator
import math
import time

color = ((255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255))
         

"""
Geometric method to locate eye region.
vertically  : middle 1/4 to 1/2 of face is eye region
horizontally: 1/6 to 3/6 left eye, 3/6 to 5/6 right eye
"""
def eye_region(x,y,w,h):
    offset = w/20
    #eye_y = np.int32(y + h/4)
    #eye_h = np.int32(h/4)
    eye_y = np.int32(y + 0.3*h)
    eye_h = np.int32(0.15*h)
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
        print "Elapsed time is " + str((time.time() - startTime_for_tictoc)*1000) + " ms."
    else:
        print "Toc: start time not set"



## Define functions
## Convert list to 1D array for numerical computation
def convert_to_array(input_array):
    xval = np.array([], dtype=np.int8)
    yval = np.array([], dtype=np.int8)
    for row in input_array:
        for x, y in row:
            xval = np.append(xval, x)
            yval = np.append(yval, y)
    return xval, yval 

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(gray, sobel_kernel=3, thresh=(-np.pi, np.pi)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    ## 3) Calculate the magnitude 
    abs_sobelx= np.absolute(sobelx)
    abs_sobely= np.absolute(sobely)
    abs_sobel = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    thresh_min = 50
    thresh_max = 200
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
       

##    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(sobely, sobelx)*180/np.pi
    # 5) Create a binary mask where direction thresholds are met
    #binary_output = dir_grad * sbinary
    # 6) Return this mask as your binary_output image
    return sbinary, dir_grad


''' multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades'''
''' https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml'''
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
''' https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml'''

cap = cv2.VideoCapture(0)  ## change to 1 if the computer has built-in camera.
#cap = cv2.VideoCapture('sample_head_fixed.avi')
#cap = cv2.VideoCapture('input5s.avi') # Not working to detect pupil for this test video


# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
out = cv2.VideoWriter('output-1.avi', -1, 20.0, (640,480))


eh=np.int32(1)
ew=np.int32(1)
skippedFrame = 0

master_counter = 0 

'''Open text file to store data'''
##print("Enter file name, eg. test-1.txt")
##str = raw_input('-->')
##text_file = open(str, "w")


# Optimization - First face detection we use Haar cascade face detector and subsequent face detection
# we apply Kalman filter for speed concern.

#ret = 1 # second condition for master while loop
#print(cap.isOpened())
while (cap.isOpened()):
    ret, img = cap.read()
    if ret == 0:
        print("No image")
        ## Press 'q' to exit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        img =cv2.flip(img,0)  ## uncomment for real-time video stream

        ## convert to HLS
        #HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        #L = HLS[:,:,1]
        
        ## Grayscale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''frameCount = frameCount + 1'''
        '''print 'frame # ', frameCount'''
        blur = gray
        filer_size = 3
        blur = cv2.medianBlur(gray, filer_size)  #median Filter

        faces = face_cascade.detectMultiScale(blur, 1.3,5) #Return multiple faces if there are. 

         

        #tic()
        if len(faces): 
            # Only when faces are detected, then proceed eye search, otherwise skip this iteration.
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
                left_eye_roi = blur[ey:ey+eh, lex:lex+ew]
                right_eye_roi = blur[ey:ey+eh, rex:rex+ew]

                left_eye_color = img[ey:ey+eh, lex:lex+ew]
                right_eye_color = img[ey:ey+eh, rex:rex+ew]
                right_eye_color_copy = np.copy(right_eye_color)
                #right_eye_color_copy = np.copy(left_eye_color)

            #right_eye_color = left_eye_color
            #im = left_eye_roi
            im = right_eye_roi
            sobel_binary, dir_grad = dir_threshold(im, sobel_kernel=3, thresh=(-np.pi/2, np.pi/2))


            '''==========================='''

            sobel_binary = sobel_binary.astype(np.uint8)
            edges = sobel_binary

            '''Canny edge'''
            thresh_high = 100#80#100 #200 #160
            thresh_low = 30#20#30 #120 #80
            edges = cv2.Canny(im,thresh_low,thresh_high)
                        
            _, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            entropy_list=np.array([], dtype=np.float)
            contours_old_index = np.array([], dtype=np.uint8)
            '''Next, we need to sample dir_grad only at contour pixels.'''
            for index, cnt in enumerate(contours):
                M = cv2.moments(cnt)
                perimeter = cv2.arcLength(cnt,True)

                '''Level-1 Criteria'''
                ## M00 (contour area) is more than 2 pixels
                ## AND 4 < perimeter < 200 pixels
                if (M['m00'] > 2) and (perimeter < 200)and(perimeter > 4):
                #if (M['m00'] > 2):
                    dir_grad_cnt=np.array([], dtype=np.float)
                    x, y = convert_to_array(cnt)
                    dir_grad_cnt = np.append(dir_grad_cnt, dir_grad[y,x])

                    # density is true to return probability
                    hist, _ = np.histogram(dir_grad_cnt, bins=np.arange(-180, 202.5, 22.5), density= False)
                    # Convert hist from number of samples to probability mass.
                    hist = np.asarray(hist, dtype=np.float)
                    hist = hist/np.sum(hist)
                    #print(hist)

                    '''Compute entropy for each cnt'''
                    entropy = 0.0 
                    for bin_probability in hist:
                        if bin_probability != 0:
                            entropy = entropy + math.log(bin_probability,2)*bin_probability
                    entropy = np.absolute(entropy)
                    entropy_list = np.append(entropy_list, entropy)
                    contours_old_index = np.append(contours_old_index, index)


            #print(np.max(entropy_list))
            ## Match index_max location to old contours index
            ## Initialize index_max
            index_max = 1024
            #print(len(entropy_list))
            if len(entropy_list)!= 0:
                index_max = np.argmax(entropy_list)
                index_max = contours_old_index[index_max]
            #print(index_max)


            '''Draw all contours'''
            survival_counter = 0 
            for index, cnt in enumerate(contours):
                # Choose gradient entropy values > 3.7
                #if entropy_list[index] > 2:
                cv2.drawContours(right_eye_color_copy, cnt, -1, color[index%6], 1)
                survival_counter = survival_counter + 1
            #print(survival_counter)         
 
            ## Draw contour with max entropy value
            if index_max != 1024: ## index_max not equal to initialized value 1024
                #cv2.drawContours(right_eye_color, contours[index_max], -1, [0,0,255], 1)
                '''Method-1 Fit ellipse'''
                ## Fit ellipse
                ## Example: (x,y),(ma,MA),angle = cv2.fitEllipse(cnt)
                ellipse = cv2.fitEllipse(contours[index_max])
                ## Draw ellipse
                cv2.ellipse(right_eye_color,ellipse,(0,255,0),1)
                #print(ellipse[1][0], ellipse[1][1], ellipse[2])#(ellipse[1][0]+ellipse[1][1])/2)

                ## Save major axis of ellipse(in pixels)
                ##timestamp=time.time()
                ##text_file.write("{:}\t{:,.6f}\n".format(timestamp, ellipse[1][1]))
                
                

##                ## Fit circle
##                (x,y),radius = cv2.minEnclosingCircle(contours[index_max])
##                center = (int(x),int(y))
##                radius = int(radius)
##                cv2.circle(right_eye_color,center,radius,(0,255,0),1)

                
                                
            cv2.imshow('Max entropy contour', right_eye_color)
            cv2.imshow('All contours', right_eye_color_copy)

            edges=edges.astype(np.float)
            cv2.imshow('canny edges', edges)
            cv2.imshow('img',img)

            #cv2.waitKey(0)


        else:
            left_eye_color=np.zeros([eh, ew, 3])
            right_eye_color=np.zeros([eh, ew, 3])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'No pupil detected',(20,320), font, 2,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow('img',img)

            
                
            ## Write 0 value to text.
            ##timestamp=time.time()
            ##text_file.write("{:}\t{:}\n".format(timestamp, 0))
            

            #cv2.waitKey(0)  ## waitKey(0) displays window infinitely until any keypress.
                             ## waitKey(25) will display a frame for 25 ms, after which display will be automatically closed.


        # write the flipped frame
        out.write(img)
        
        master_counter = master_counter + 1
        #toc()
        
        # Press 'q' to exit program anytime in while loop. 
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


#print('Exit while loop')
#text_file.close()
if cap.isOpened() == True:
    cap.release()

out.release()

cv2.waitKey(0)
cv2.destroyAllWindows()





