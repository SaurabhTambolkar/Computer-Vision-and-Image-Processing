import numpy as np

def findRotMat(a,b,g):
    #converting to radians
    a = np.radians(a)
    b = np.radians(b)
    g = np.radians(g)

    #rotate around Z in 45 degrees
    rZ1 = np.array(((np.cos(a), -np.sin(a), 0),(np.sin(a), np.cos(a), 0),(0,0,1)))

    #rotate around X in 30 degrees
    rX = np.array(((1,0,0),(0, np.cos(b), -np.sin(b)),(0, np.sin(b), np.cos(b))))

    #rotate at Z in 60 degrees
    rZ2 = np.array(((np.cos(g), -np.sin(g), 0),(np.sin(g), np.cos(g), 0),(0,0,1)))
    
    #rX * rY
    s = np.matmul(rZ2,rX)

    #rX* rY * rZ
    xyzToXYZ = np.matmul(s,rZ1)
   

    # Inverse Matrix
    XYZToxyz = xyzToXYZ.transpose()

    return xyzToXYZ,XYZToxyz



alpha = 45
beta = 30
gamma = 60
rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
print("Rotation Matrix from xyz to XYZ ")
print(rotMat1)
print("Rotation Matrix from XYZ to xyz ")
print(rotMat2)
print("*****************************")





    
    
