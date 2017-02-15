# -*- coding: utf-8 -*-
from Filters import KalmanFilter
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

class MovingObject(object):
    
    idx = 0
    
    def __init__(self, box):
        
        self.id = MovingObject.idx
        self.box = box
        self.age = 1
        self.totalVisibleCount = 1
        self.consecutiveInvisibleCount = 0
        self.track= [box]
        self.time = time.time()
        
        #Set Filter Information
        delT = 0.0
        A = np.matrix([[1,delT,0,0],[0,1,0,0],[0,0,1,delT],[0,0,0,1]]) #State Transition
        B = np.matrix(np.zeros((4,4)))
        H = np.matrix([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
        x,y = box
        u,v = (0.,0.)
        X = np.matrix([[x],[u],[y],[v]]) #Initial State Matrix
        P = np.matrix(200*np.eye(4))
        Q = np.matrix(200*np.eye(4))
        R = np.matrix(100*np.eye(4))
        self.filter = KalmanFilter(A,B,H,X,P,Q,R)
        MovingObject.idx+=1

    def addTrack(self, box):
        self.track+=box
    
    def predictBoxLocation(self):
        delT = time.time() - self.time
        self.time += delT
        
        self.filter.A = np.matrix([[1,delT,0,0],[0,1,0,0],[0,0,1,delT],[0,0,0,1]]) #State Transition
        controlVector = np.matrix(np.zeros((4,1)))
        self.filter.predict(controlVector)
        
        return self.filter.X
        
    def observeUpdate(self, box):
        x,y = box
        measurementVector = np.matrix([[x],[0.],[y],[0.]])
        self.filter.observe(measurementVector)
        self.filter.update()
        
def euclidean(a,b):
    x1,y1 = a
    x2,y2 = b
    distance = ((x2-x1)**2.0 + (y2-y1)**2.0)**0.5
    return distance
    
def assign(tracks, boxes):
    n = len(tracks)
    m = len(boxes)
    cost = np.matrix(np.zeros((n,m)))
    
    for i in range(n):
        for j in range(m):
            locationX = tracks[i].predictBoxLocation()[0].item()
            locationY = tracks[i].predictBoxLocation()[2].item()
            location = (locationX,locationY)
            cost[i,j] = euclidean(location, boxes[j])
    
    row_ind, col_ind = linear_sum_assignment(cost)
    return (row_ind, col_ind, cost)
    
trackSet = []
t = [[(1,1), (100,1)], [(1,2)], [(100,3)], [(1,4),(100,4)], [(1,5),(100,5)], [(1,6), (100,6)]]
for boxes in t:
    time.sleep(1)
    if not len(trackSet):
        for box in boxes:
            obj = MovingObject(box)
            trackSet.append(obj)
            
    else:
        index = assign(trackSet, boxes)
        row, col, cost = index
        for r,c in zip(row, col):
            trackSet[r].observeUpdate(boxes[c])



