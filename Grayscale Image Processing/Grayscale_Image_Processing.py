"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolve2d. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolve2d or correlation. 
You should NOT use any other libraries, which provide APIs for convolve2d/correlation ormedian filtering. 
Please write the convolve2d code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
from numpy import divide, int8, multiply, ravel, sort, zeros_like
from math import sqrt
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """
    mask = 3
    
    bd = int(mask / 2)
    gray_img = img.copy()
    
    denoise_img = zeros_like(gray_img)
    
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
        
            # get mask according with mask
            kernel = ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])
            
            # calculate mask median
            median = sort(kernel)[int8(divide((multiply(mask, mask)), 2) + 1)]
            
            denoise_img[i, j] = median
            
    return denoise_img


def AddZeroPadding(img):

    padded_img = img.copy()
    
    padded_img = np.pad(padded_img, ((1,1),(1,1)), 'constant')
    return padded_img
    
def FlipKernel(kernel):

    k = kernel.copy()
    
    k1 = np.zeros((3,3)).astype(int)
    
    for i in range(0,3):
        for j in range(0,3):       
            k1[i][j] = int(k[2 - i][2 - j])
    
    #print(kernel,k1)
    return k1

def convolve2d(img, kernel):
    """
    :param im: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    h = len(img)
    w = len(img[0])

    h1 = h-3
    w1 = w-3
    conv_img = np.zeros((h1,w1))

    for i in range(0,h1):
        for j in range(0,w1):
            temp_matrix = img[i:i+3,j:j + 3]
            
            conv_img[i,j] = np.sum(np.multiply(temp_matrix,kernel))
            
    return conv_img


def edge_magnitude(x,y):
    h,w = x.shape
    new_img = np.zeros((h,w))
    for i in range(0,h):
        for j in range(0,w):
            x2 = x[i,j] * x[i,j]
            y2 = y[i,j] * y[i,j]
            sum1 = x2 + y2
            new_img[i,j] = sqrt(sum1)
    return new_img
    
def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    im1 = img.copy()
    im1 = AddZeroPadding(im1)
    X = convolve2d(im1,FlipKernel(sobel_x))
    edge_x = 255 * (X - X.min())/(X.max() - X.min())
    Y = convolve2d(im1,FlipKernel(sobel_y))
    edge_y = 255 * (Y - Y.min())/(Y.max() - Y.min())
    mag = edge_magnitude(X,Y)
    edge_mag = 255 * (mag - mag.min())/(mag.max() - mag.min())
    
    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    im1 = img.copy()
    
    #Edge Detection for edge_45 kernel
    Kernel45 = np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]]).astype(int)
    edge_45 = convolve2d(im1,Kernel45)
    edge_45 = 255 * (edge_45 - edge_45.min())/(edge_45.max() - edge_45.min())
    
    #Edge Detection for edge_135 kernel
    Kernel135 = np.array([[ 2,  1,  0], [1,   0,  -1], [0,  -1, -2]]).astype(int)
    edge_135 = convolve2d(im1,Kernel135)
    edge_135 = 255 * (edge_135 - edge_135.min())/(edge_135.max() - edge_135.min())
    
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)





