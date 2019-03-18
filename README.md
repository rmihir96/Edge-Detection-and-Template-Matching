### Problem Definition:
UB-CSE573 Computer Vision and Image Processing Project 1.

Given a set of images(found in the folder named "data") perfom two tasks:
1. Edge Detection (proj1-task1.jpg)
2. Template Matching (proj1-task2.jpg)
The task is to be perfomed without using library functions from cv2. 
The logic for edge detection and template matching is to be written in Python from scratch. 
Helper functions are provided in utils.py to assist in completing the tasks at hand.


### Edge Detection(task1.py)


Kernel Opertors used : Sobel and Prewitt.

Calculate gradient along X axis and along Y axis for both operators.
Then calculate the magnitude of gradient along both axes which is the final resultant image with detected edges.
Convolve the image with the given kernel to get output image.


### Template Matching(task2.py)

Find characters "a", "b" and "c" from the given image using templates in the folder named "data".
Perfom normalized cross correlation of the image and the template.

