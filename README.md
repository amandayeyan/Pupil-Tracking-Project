#Face-Eye-Pupil-Tracking
##Setups	
- Software: Windows 8, OpenCV 3.10 and Pyhton 2.7	
- Hardware: Logitech webcam C525

##1. Simple way to realise real-time face tracking
Use frontal face template
- You may take a snapshot of your frontal face using webcam and crop out only the face region (squared region) without background. 
- Name the cropped image as 'myFace-template.jpg'.


Try tune the threshold value to get more robust result. 

Even when you turn your head a little bit, it can trace your face with low threshold value.	Change to another face, it still works.


##2. Apply Haar Cascades frontal face detector 
Download xml file from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 
Note the license using the Cascades.

##3. Apply canny edge and Hough Transform to find circles in extracted eye region. 
So far,a complete version is realised but can only track full circles.  


Coming soon ...
