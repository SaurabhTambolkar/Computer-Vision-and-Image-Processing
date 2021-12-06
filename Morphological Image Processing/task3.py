"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

import numpy as np
from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows

struct_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(int)

def morph_dilate(im):
    """
    :param im: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
    img = im.copy()
    w, h = img.shape
    
    res = np.asarray([[0 for _ in range(h)] for _ in range(w)])
    
    # Iterating the anchor of the structure image over the original image
    for i, im_row in enumerate(img):
        for j, im_ele in enumerate(im_row):
            if img[i][j]:
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if (i+k) >= w or (i+k) < 0 or (l+j) >= h or (l+j) < 0:
                            continue
                        else:
                            res[i+k][j+l] = 255

    return res

def morph_erode(im):

    img = im.copy()
    width, height = img.shape
    se_w, se_h = struct_element.shape
    
    res = np.asarray([[0 for _ in range(height)] for _ in range(width)])
    
    for i, im_row in enumerate(img):
        for j, im_ele in enumerate(im_row):
            Temp_Flag = True
            
            for k in range(-1, 2):
                for l in range(-1, 2):
                
                    if (i+k) >= width or (i+k) < 0 or (l+j) >= height or (l+j) < 0:
                        continue
                    if struct_element[k][l] != 0 and img[i+k][j+l] == 0:
                        Temp_Flag = False
            if Temp_Flag:
                res[i][j] = 255

    return np.asarray(res)


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    new_img = img.copy()
    
    temp = morph_erode(new_img)
    open_img = morph_dilate(temp)
    
    return open_img

    
def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    new_img = img.copy()
    
    temp = morph_dilate(new_img)
    close_img = morph_erode(temp)
    
    return close_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    new_img = img.copy()
    
    ToSubtract = morph_erode(new_img)
    bound_img = np.subtract(new_img, ToSubtract)
    
    return bound_img
    
    
def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    new_img = img.copy()
    
    temp1 = morph_open(new_img)
    denoise_img = morph_close(temp1)
    
    return denoise_img


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)





