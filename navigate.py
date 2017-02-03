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
def getGoodMatches(matches, kp1, kp2, des1, des2, epipolar=False):
    
    good = []
    for m,n in matches:
        if m.distance < 1*n.distance:
            good.append(m) 
    if epipolar:
        matches = []
        for m in good:
            trainIdx = m.trainIdx
            queryIdx = m.queryIdx
            if int(kp1[trainIdx].pt[1]) == int(kp2[queryIdx].pt[1]):
                matches.append(m)
        good = matches
        
    return good

def getMapping(matches, kp1, kp2):
    feature = {}
    for m in matches:
        trainIdx = m.trainIdx
        queryIdx = m.queryIdx
        feature[kp1[trainIdx]] = kp2[queryIdx]
    return feature
    
data = Dataset(True)
img = np.zeros((5000, 6000, 3), dtype='uint8')
offset = [3000, 230, 180]
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

focalX = 718.856
focalY = 718.856
Cx = 607.192
Cy = 185.2157
baseline = 0.54

mtx = np.matrix([[focalX, 0, Cx],[0, focalY, Cy],[0, 0, 1]], dtype='float32')
cam = Camera(mtx, baseline)
pose = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype='float32')

cam.setPose(pose)
transformation = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype='float32')
orb = cv2.ORB_create()
bf = cv2.BFMatcher()
while True:
    
    ret = data.iterate()
    if not ret:
        break
    left, right, left2, right2 = data.getImageStereo()
    pose = data.getPose()
    pose = np.vstack((pose, np.matrix([0,0,0,1])))
    kp1, des1 = orb.detectAndCompute(left, None)
    kp2, des2 = orb.detectAndCompute(right, None)
    kp3, des3 = orb.detectAndCompute(left2, None)
    kp4, des4 = orb.detectAndCompute(right2, None)
    
    matches = bf.knnMatch(des2, des1, k=2)    
    matchesL2R = getGoodMatches(matches, kp1, kp2, des1, des2, True)    
    
    matches = bf.knnMatch(des4, des2, k=2)    
    matchesR2R = getGoodMatches(matches, kp2, kp4, des2, des4)
    
    matches = bf.knnMatch(des3, des4, k=2)    
    matchesR2L = getGoodMatches(matches, kp4, kp3, des4, des3, True)
    
    matches = bf.knnMatch(des1, des3, k=2)    
    matchesL2L = getGoodMatches(matches, kp3, kp1, des3, des1)
    
    featuresL2R = getMapping(matchesL2R, kp1, kp2)
    featuresR2R = getMapping(matchesR2R, kp2, kp4)
    featuresR2L = getMapping(matchesR2L, kp4, kp3)
    featuresL2L = getMapping(matchesL2L, kp3, kp1)
    features_prev = []
    features_curr = []
    for m in matchesL2R:
        
        try:
            if int(featuresL2L[featuresR2L[featuresR2R[featuresL2R[kp1[m.trainIdx]]]]].pt[0])  == int(kp1[m.trainIdx].pt[0]) and int(featuresL2L[featuresR2L[featuresR2R[featuresL2R[kp1[m.trainIdx]]]]].pt[1])  == int(kp1[m.trainIdx].pt[1]):   
                
                features_prev.append((kp1[m.trainIdx],featuresL2R[kp1[m.trainIdx]]))
                features_curr.append((featuresR2L[featuresR2R[featuresL2R[kp1[m.trainIdx]]]],featuresR2R[featuresL2R[kp1[m.trainIdx]]]))
        except:
            pass
    
    X= np.array(Triangulate(features_prev), dtype='float32')
    X2 = np.array(Triangulate(features_curr), dtype='float32')
    
    F = []
    for f1, f2, in features_curr:
        F.append((f2.pt))
    F = np.array(F, dtype='float32')
    
    ret, rvec, tvec = cv2.solvePnP(X, F, cam.getCalibrationMatrix(), None)
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec
    #translation = rotation.dot(tvec)
    transformation = np.hstack((rotation, translation))
    transformation = np.vstack((transformation, np.matrix([0,0,0,1], dtype='float32')))
    cam.setPose(cam.getPose().dot(transformation))
    x = cam.getPose()[0,3]
    y = cam.getPose()[1,3]
    z = cam.getPose()[2,3]
    
    
    img = cv2.circle(img, (int(x*10) + offset[0], int(z*10) + offset[2]), 50, (0,0,255), -1)
    x = data.getPose()[0,3]
    y = data.getPose()[1,3]
    z = data.getPose()[2,3]
    
    img = cv2.circle(img, (int(x*10) + offset[0], int(z*10) + offset[2]), 50, (0,255,0), -1)
    cv2.imshow("Frame", img)
    cv2.imshow("Frame2", left)
    #cv2.imshow("Frame2", left)
    ch = cv2.waitKey(0)
    if ch == ord('q'):
        break
    #print ("Running..")
cv2.destroyAllWindows()