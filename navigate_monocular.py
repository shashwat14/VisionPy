from Dataset import Dataset, Geometry, Camera
import cv2
import numpy as np

data = Dataset(True)
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
left, right, left2, right2 = data.getImageStereo()

fast = cv2.FastFeatureDetector_create()
fast.setType(2)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 15, 0.01))
rotation = np.matrix([[1,0,0],[0,1,0],[0,0,1]], dtype='float32')
trans = np.matrix([0,0,0])

while ret:
    
    kp = fast.detect(left,None)
    
    p0 = [] 
    for k in kp:
        p0.append([[k.pt[0],k.pt[1]]])
    p0 = np.array(p0, dtype='float32')
    p1, st, err = cv2.calcOpticalFlowPyrLK(left, left2, p0, None, **lk_params)
    
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    E = cv2.findEssentialMat(good_new, good_old, cam.getCalibrationMatrix(),cv2.RANSAC, 0.999, 1.0, None)
    inliers, rvec, tvec, mask = cv2.recoverPose(E[0], good_new, good_old, cam.getCalibrationMatrix())
    
    currT = data.getPose()[:,-1]
    pastT = posePast[:,-1]
    scale = ((pastT[0]-currT[0])**2 + (pastT[1]-currT[1])**2 + (pastT[2]-currT[2])**2).item()**0.5
    posePast = data.getPose()
    
    if scale > 0.1 and tvec[2].item() > tvec[0].item():
        trans = trans + scale*tvec.T.dot(rotation)
        rotation = rvec.dot(rotation)
    
    x1 = -trans[0,0]
    y1= trans[0,1]
    z1 = trans[0,2]
    cv2.circle(img, (int(x1)+offset[0], int(z1)+offset[2]), 1, (0,0,255), -1)
    
    
    
    
    left, right, left2, right2 = data.getImageStereo()
    pose = data.getPose()
    x2 = pose[0,-1]
    y2 = pose[1,-1]
    z2 = pose[2,-1]
    cv2.circle(img, (int(x2)+offset[0], int(z2)+offset[2]), 1, (0,255,0), -1)
    pose = np.vstack((pose, np.matrix([0,0,0,1])))
    error = ((x2-x1)**2. + (z2-z1)**2.)**0.5
    print error
    cv2.imshow("Frame", img)
    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break
    
    ret = data.iterate()
cv2.destroyAllWindows()


