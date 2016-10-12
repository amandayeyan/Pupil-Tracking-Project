# Face-Eye-Pupil-Tracking
Setups	
- Software: Windows 8, OpenCV 3.10 and Pyhton 2.7	
- Hardware: Logitech webcam C525

1(a). Simple way to realise real-time face tracking   
using frontal face template
- You may take a snapshot of your frontal face using webcam and crop out only the face region (squared region) without background. 
- Name the cropped image as 'myFace-template.jpg'.


Try tune the threshold value to get more robust result. 

Even when you turn your head a little bit, it can trace your face with low threshold value.	Change to another face, it still works.
