import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners
    
Crit = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001) #Setting criteria for accuracy


def FindIntrinsicParam(MatrixM):
    ox = np.sum(np.multiply(MatrixM[0,0:3], MatrixM[2,0:3]))
    oy = np.sum(np.multiply(MatrixM[1,0:3], MatrixM[2,0:3]))
    
    fx = np.sqrt(np.sum(MatrixM[0,0:3]**2) - ox**2)
    fy = np.sqrt(np.sum(MatrixM[1,0:3]**2) - oy**2)
    
    return [fx,fy,ox,oy]
    
    
def calibrate(imgname):

    ObjectPoints = [] # points in the real world space
    ImagePoints = [] # points in a image plane.

    objp = np.array([[4,0,4],[4,0,3],[4,0,2],[4,0,1],
                           [3,0,4],[3,0,3],[3,0,2],[3,0,1],
                           [2,0,4],[2,0,3],[2,0,2],[2,0,1],
                           [1,0,4],[1,0,3],[1,0,2],[1,0,1],
                           [0,1,4],[0,1,3],[0,1,2],[0,1,1],
                           [0,2,4],[0,2,3],[0,2,2],[0,2,1],
                           [0,3,4],[0,3,3],[0,3,2],[0,3,1],
                           [0,4,4],[0,4,3],[0,4,2],[0,4,1]],
                           np.float32)
    
    #multiplying each point by 10
    objp *= 10 #The edge length of each grid on the checkerboard is 10mm
    
    img = imread(imgname)
  
    gray = cvtColor(img,COLOR_BGR2GRAY)
    
    found, corners1=findChessboardCorners(gray,(4,9),None)
 
    if found==True:                

        #Only 32 points needed
        #Deleting points on Y Axis
        corners1  = np.delete(corners1, tuple(range(16,20)), axis=0)
        
        corners2 = cornerSubPix(gray, corners1,(4,4),(-1,-1), Crit)
        corners2 = cornerSubPix(gray, corners1,(4,4),(-1,-1), Crit)
        
        
        ImagePoints.append(corners2) 
        ObjectPoints.append(objp)
        
        img = drawChessboardCorners(img, (4,4), corners2[0:15,:,:],found)
        img = drawChessboardCorners(img, (4,4), corners2[16:31,:,:],found)
        
        
    worldpoints = ObjectPoints[0]
    imagepoints = ImagePoints[0]
    imagepoints = imagepoints.reshape((imagepoints.shape[0] * imagepoints.shape[1]), imagepoints.shape[2])
    
    wrow, wcol = worldpoints.shape
    irow, icol = imagepoints.shape
     
    PM = np.zeros((2*wrow,12))
    
    for i in range(wrow):
        
        X,Y,Z = worldpoints[i]
        x,y = imagepoints[i]
        
        PM[2*i] = np.array([  X,  Y,  Z,  1,  0,  0,  0,  0, -(X*x), -(Y*x), -(Z*x), -(x)])
        PM[(2*i) + 1] = np.array([  0,  0,  0,  0,  X,  Y,  Z,  1, -(X*y), -(Y*y), -(Z*y), -(y)])
    
    U, S, VT = np.linalg.svd(PM)

    #creating lamda for M Matrix Creation
    xroots = VT[-1]
    lamda = np.sqrt(1 / np.sum(xroots[8:11]**2))
    
    MatrixM = lamda * xroots    
    MatrixM = MatrixM.reshape(3,4)
    
    return FindIntrinsicParam(MatrixM), True


#Calling calibrate Method
Intinsic_Matrix_parameters, is_constant = calibrate('checkboard.png')

print("fx : ",Intinsic_Matrix_parameters[0])
print("fy : ",Intinsic_Matrix_parameters[1])
print("Ox : ",Intinsic_Matrix_parameters[2])
print("Oy : ",Intinsic_Matrix_parameters[3])

if is_constant == True:
    print("The intrinsic parameters are invariable.")
else:
    print("The intrinsic parameters differed from world coordinates.")

