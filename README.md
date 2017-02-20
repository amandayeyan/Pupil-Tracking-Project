#Pupil-Tracking-Project
This project aims to track the pupil boundary in real-time.

##Setups	
- Software: Windows 8, OpenCV 3.10 and Pyhton 2.7	
- Hardware: IR camera, resolution 640x480.

##Pipeline
### Pre-process to crop eye region
1. Read video streaming from camera frame by frame
2. Grayscale
3. Smooth image with median blur filter. For this application, median blur works better than gaussian blur.
4. Apply Haar Cascades frontal face detector[1]
5. Crop eye region with geometry assumption that eye location is relatively fixed to face.

### Apply algorithms to fit pupil boundary with ellipse (Demo for one pupil only)
4. Compute sobel directions of gradient
5. Canny edge
6. Find contours on the canny edges. Discard contours with large arc perimeter.
7. Compute gradient entropy (refer to Cihan Topal's publication[2]).
   Select maximum entropy among all contour candidates.
8. Fit ellipse for the contour


Reference
[1] Apply Haar Cascades frontal face detector 
Download xml file from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 
Note the license using the Cascades.
[2] C.Topal and C. Akinlar, An Adaptive Algorithm for Precise Pupil Boundary Detection Using the Entropy of Contour Gradients.2013. Elsevier preprint.
