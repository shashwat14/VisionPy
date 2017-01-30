from Dataset import Dataset, Geometry
import cv2
import numpy as np
data = Dataset(True)
img = np.zeros((5000, 3000, 3), dtype='uint8')
maxX = 0
maxY = 0
maxZ = 0
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
while True:
    data.iterate()
    left, right, left2, right2 = data.getImageStereo()
    pose = data.getGroundTruth()
    x = pose[0,3]
    y = pose[1,3]
    z = pose[2,3]
    if abs(x) > maxX:
        maxX = abs(x)
    if abs(y) > maxY:
        maxY = abs(y)
    if abs(z) > maxZ:
        maxZ = abs(z)
    cv2.circle(img,(abs(int(x*10)),abs(int(z*10))),10,(0,255,0), -1)
    cv2.imshow("Frame", img)
    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break
    print ("Running..")
cv2.destroyAllWindows()