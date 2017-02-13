# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:12:28 2017

@author: Shashwat
"""
import numpy as np
import cv2

class Dataset(object):

    def __init__(self, flag=False):
        self.id = -1
        with open('/home/therumsticks/Downloads/00/poses/00.txt', 'rb') as f:
            content = f.readlines()
        for  i in range(len(content)):
            content[i] = content[i].decode('utf-8').strip().split()
            content[i] = [float(x) for x in content[i]]
            content[i] = np.matrix(content[i], dtype='float32').reshape(3,4)
        self.content = content
        self.getTwoFrames = flag
        
        
    def getPose(self, i=0):
        return self.content[self.id + i]
    
    def getImageStereo(self):
        i= self.id
        if i < 10:
            imgURL = "00000" + str(i)
            if i == 9:
                imgURL2 = "000010"
            else:
                imgURL2 = "00000" + str(i+1)
        elif i < 100:
            imgURL = "0000" + str(i)
            if i == 99:
                imgURL2 = "000100"
            else:
                imgURL2 = "0000" + str(i+1)
        elif i < 1000:
            imgURL = "000" + str(i)
            if i == 999:
                imgURL2 = "001000"
            else:
                imgURL2 = "000" + str(i+1)
        elif i < 10000:
            imgURL = "00" + str(i)
            if i == 9999:
                imgURL2 = "010000"
            else:
                imgURL2 = "00" + str(i+1)
            
        #Left image
        leftURL = "/home/therumsticks/Downloads/00/image_0/" + imgURL + ".png"
        left = cv2.imread(leftURL, 0)
        
        #right image
        rightURL = "/home/therumsticks/Downloads/00/image_1/" + imgURL + ".png"
        right = cv2.imread(rightURL, 0)
        
        #left image at t + 1
        leftURL2 = "/home/therumsticks/Downloads/00/image_0/" + imgURL2 + ".png"
        left2 = cv2.imread(leftURL2, 0)
        
        #right image at t + 1
        rightURL2 = "/home/therumsticks/Downloads/00/image_1/" + imgURL2 + ".png"
        right2 = cv2.imread(rightURL2, 0)
        if self.getTwoFrames:
            return (left, right, left2, right2)
        else:
            return (left, right)
        
    def getImage(self):
        pass
    def iterate(self):
        if self.id == 4538:
            return False
        else:
            self.id+=1
            return True

class Geometry(object):
    
    def __init__(self):
        pass
class Camera(object):
    
    def __init__(self, mtx, baseline = 1.):
        self.mtx = mtx
        self.baseline = baseline
    def getCalibrationMatrix(self):
        return self.mtx
    def setPose(self, pose):
        self.pose = pose
    def getPose(self):
        return self.pose
    def move(self, mtx):
        self.move = self.move.dot(mtx)
    def triangulate(self, x1, y1, x2, y2):
        x1 = x1 - self.mtx[0,2]
        x2 = x2 - self.mtx[0,2]
        y1 = y1 - self.mtx[1,2]
        Z = self.mtx[0,0]*self.baseline/(x1-x2)
        
        X = x1*Z/self.mtx[0,0]        
        Y = y1*Z/self.mtx[0,0]        
        return np.matrix([X,Y,Z])
        