# -*- coding: utf-8 -*-

from Dataset import Dataset, Camera
import cv2
import numpy as np
from Geometry import Point3D

def Triangulate(correspondences):
    left, right = correspondences
    P = []
    for l, r in zip(left,right):
        if int(l[1]) == int(r[1]):
            X = robot.triangulate(l[0],l[1],r[0],r[1])
            P.append(Point3D(data.id, X, l))
    return P
        
        
def detectFeaturesAndCorrespondence(left, right):
    kp = fast.detect(pleft, None)
    p0 = []
    for k in kp:
        p0.append([[k.pt[0],k.pt[1]]])
    p0 = np.array(p0, dtype='float32')
      
    p1, st, err = cv2.calcOpticalFlowPyrLK(pleft, pright, p0, None, **lk_params)
    good_right = p1[st==1]
    good_left = p0[st==1]
    
    return (good_left, good_right)
    
    

#Set intiial values
Graph = []
data = Dataset()
focalX = 718.856
focalY = 718.856
Cx = 607.192
Cy = 185.2157
baseline = 0.54
mtx = np.matrix([[focalX, 0, Cx],[0, focalY, Cy],[0, 0, 1]], dtype='float32')
robot = Camera(mtx, baseline)
pose = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype='float32')
robot.setPose(pose)

ret = data.iterate()
pleft, pright = data.getImageStereo()

fast = cv2.ORB_create(1000)
lk_params = dict( winSize  = (21,21), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 0.01))

while ret:
    cv2.imshow("Frame", pleft)
    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break    
    
    #Get new images
    ret = data.iterate()
    left, right = data.getImageStereo()
    correspondences_spatio = detectFeaturesAndCorrespondence(pleft, pright)
    #Check for repeated 3D points
    Points3D = Triangulate(correspondences_spatio)
    
    correspndences_temporal = detectFeaturesAndCorrespondence(pleft, left)
    
    #Store the new images as previous images
    pleft = left.copy()
    pright = right.copy()
cv2.destroyAllWindows()
