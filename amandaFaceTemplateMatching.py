import cv2
import numpy as np

# Acquire images from real-time webcam video. 
cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your own frontal face template to try
    template = cv2.imread('myFace-template.jpg')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w,h = template_gray.shape

    # Multiple match algorithm shown here.
    # http://docs.opencv.org/3.1.0/d4/dc6/tutorial_py_template_matching.html 
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Tune threshold value. I tried another person's face(sitting in front of webcam), it also works.
    threshold = 0.5
    cv2.imshow('Live res', res)
    loc = np.where(res >= threshold)

    # using zip to iterate over two lists in parallel
    for pt in zip(*loc[::-1]):
        #In image space, pairs always in row(y) and col(x), so pt is(y,x)
        #Draw rectangle on original image
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

    cv2.imshow('detected', img)

    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break  #Press 'ESC' key to break
    
cv2.destroyAllWindows()
cap.release()
