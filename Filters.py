import numpy as np
class KalmanFilter(object):
    
    def __init__(self, A, B, H, X, P, Q, R):
        
        self.A = A    #State Transition Matrix
        self.B = B    #Control Matrix
        self.H = H    #Observation Matrix
        self.X = X    #Initial State Estimate
        self.P = P    #Intial Covariance Estimate
        self.Q = Q    #Estimated Process Noise
        self.R = R    #Estimated Measurement Noise
    
    def predict(self, controlVector):
        self.X = self.A*self.X + self.B*controlVector
        self.P = self.A*self.P*self.A.T + self.Q
        
    def observe(self, measurementVector):
        self.innovation = measurementVector - self.H*self.X
        self.innovation_covariance = self.H*self.P*self.H.T + self.R
        
    def update(self):
        kalmanGain = self.P*self.H.T*np.linalg.inv(self.innovation_covariance)
        self.X = self.X + kalmanGain*self.innovation
        self.P = (np.eye(self.H.shape[0]) - kalmanGain*self.H)*self.P
    
    def getCurrentState(self):
        return self.X
    

A = np.matrix([[1.]])
H = np.matrix([[1.]])
Q = np.matrix([[0.0001]])
R = np.matrix([[0.1]])
X = np.matrix([[100.]])
P = np.matrix([[1.]])

B = np.matrix([1.])
obj = KalmanFilter(A, B, H, X, P, Q, R)