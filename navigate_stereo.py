from Dataset import Dataset, Geometry, Camera
import cv2
import numpy as np

def Triangulate(good_old, good_new):
    
    x1, y1 = tuple(good_old)
    x2, y2 = tuple(good_new)
    X = cam.triangulate(x1, y1, x2, y2)[:3]
    return X

data = Dataset()
img = np.zeros((500, 600, 3), dtype='uint8')
offset = [300, 23, 18]
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



posePast = data.getPose()
ret = data.iterate()
pleft, pright = data.getImageStereo()

fast = cv2.ORB_create(1000)
#fast.setType(2)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 0.01))
rotation = np.matrix([[1,0,0],[0,1,0],[0,0,1]], dtype='float32')
trans = np.matrix([0,0,0])
# -*- coding: utf-8 -*-
while ret:
    
    kp = fast.detect(pleft, None)
    ret = data.iterate()
    left, right = data.getImageStereo()
    
    p0 = []
    for k in kp:
        p0.append([[k.pt[0],k.pt[1]]])
    p0 = np.array(p0, dtype='float32')
      
    p1, st, err = cv2.calcOpticalFlowPyrLK(pleft, pright, p0, None, **lk_params)    
    
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    TracksLeft = []
    TracksLeftNew = []
    distance = []
    for i in range(len(good_new)):
        if int(good_new[i][1]) == int(good_old[i][1]) and good_new[i][0] > 0 and good_new[i][1] > 0:
            TracksLeft.append([good_old[i]])
            distance.append(Triangulate(good_old[i], good_new[i]))
            TracksLeftNew.append([good_new[i]])
    TracksLeft = np.array(TracksLeft, dtype='float32')
    TracksLeftNew = np.array(TracksLeftNew, dtype='float32')
    p2, st, err = cv2.calcOpticalFlowPyrLK(pleft, left, TracksLeft, None, **lk_params)
    
    correctStuff = []
    p =[]
    X = []
    for i in range(len(TracksLeft)):
        if st[i].item() == 1 and p2[i][0][0] > 0 and p2[i][0][1] > 0 and p2[i][0][0] < 1241 and p2[i][0][1] < 376:
            X.append(distance[i])
            p.append(p2[i])
            correctStuff.append([TracksLeft[i], p2[i], distance[i]])
            
    X = np.array(X, dtype='float32').reshape(len(X), 1, 3)
    p = np.array(p, dtype='float32')
    
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(X, p, cam.getCalibrationMatrix(), None, reprojectionError= .08, confidence=0.999, iterationsCount=200)
    pose = np.hstack((np.matrix(cv2.Rodrigues(rvec)[0]),tvec))
    pose = np.vstack((pose, [0,0,0,1]))
    if tvec[0] < abs(tvec[2]) and tvec[1] < abs(tvec[2]):
        cam.setPose(cam.getPose().dot(pose))
    
    #-----
    x1 = cam.getPose()[0,-1]
    y1= cam.getPose()[1,-1]
    z1 = -cam.getPose()[2,-1]
    cv2.circle(img, (int(x1)+offset[0], int(z1)+offset[2]), 1, (0,0,255), -1)

    pose = data.getPose()
    x2 = pose[0,-1]
    y2 = pose[1,-1]
    z2 = pose[2,-1]
    cv2.circle(img, (int(x2)+offset[0], int(z2)+offset[2]), 1, (0,255,0), -1)
    #-----
    #reinitialization
    pleft = left
    pright = right
    cv2.imshow("Frame", img)
    cv2.imshow("Frame2", left)
    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break
cv2.destroyAllWindows()