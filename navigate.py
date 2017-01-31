# -*- coding: utf-8 -*-

from Dataset import Dataset, Geometry, Camera
import cv2
import numpy as np

def Triangulate(features):
    X =[]
    for f1,f2 in features:
        
        x1, y1 = f1.pt
        x2, y2 = f2.pt        
        X.append(cam.triangulate(x1, y1, x2, y2))
    return X
data = Dataset(True)
img = np.zeros((5000, 6000, 3), dtype='uint8')
offset = [3000, 230, 180]
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

focalX = 718.856
focalY = 718.856
Cx = 607.192
Cy = 185.2157
baseline = 0.54

mtx = np.matrix([[focalX, 0, Cx],[0, focalY, Cy],[0, 0, 1]], dtype=float)
cam = Camera(mtx, baseline)
pose = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

cam.setPose(pose)

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher()
while True:
    
    ret = data.iterate()
    if not ret:
        break
    left, right, left2, right2 = data.getImageStereo()
    pose = data.getGroundTruth()
    
    kp1, des1 = orb.detectAndCompute(left, None)
    kp2, des2 = orb.detectAndCompute(right, None)
    matches = bf.knnMatch(des2, des1, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    features = []
    matches = []
    desc= []
    for m in good:
        trainIdx = m[0].trainIdx
        queryIdx = m[0].queryIdx
        if int(kp1[trainIdx].pt[1]) == int(kp2[queryIdx].pt[1]):
            features.append((kp1[trainIdx],kp2[queryIdx]))
            desc.append((des1[trainIdx],des2[queryIdx]))
            matches.append(m[0])
    X = Triangulate(features)
    img3 = cv2.drawMatches(right, kp2, left, kp1, matches, None, flags=2)
    cv2.imshow("Frame", img3)
    #cv2.imshow("Frame2", left)
    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break
    print ("Running..")
cv2.destroyAllWindows()