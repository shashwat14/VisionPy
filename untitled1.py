# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:52:14 2017

@author: therumsticks
"""
from Dataset import Dataset, Geometry, Camera
import numpy as np
import cv2
def pp(point):
    global cam
    point = cam.getCalibrationMatrix().dot(point)
    return (point/point[-1])[:2]
    
focalX = 718.856
focalY = 718.856
Cx = 607.192
Cy = 185.2157
baseline = 0.54

mtx = np.matrix([[focalX, 0, Cx],[0, focalY, Cy],[0, 0, 1]], dtype='float32')
cam = Camera(mtx, baseline)

points = []
point = np.matrix([1,2,5]).T
points.append(point)
point = np.matrix([1,0,10]).T
points.append(point)
point = np.matrix([1,1,20]).T
points.append(point)
point = np.matrix([0,2,10]).T
points.append(point)


points2d = []
for p in points:
    points2d.append(pp(p))


for i in range(len(points)):
    points[i] = points[i] - np.matrix([0,0,1]).T

newPoints = []
for p in points:
    newPoints.append(pp(p))    

for i in range(len(points)):
    points[i] = points[i] + np.matrix([0,0,1]).T

points = np.array(points, dtype='float32')
newPoints = np.array(newPoints, dtype='float32')

ret , rvec, tvec = cv2.solvePnP(points, newPoints, cam.getCalibrationMatrix(), None)
