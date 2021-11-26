
"""
Created on Wed Nov  24 13:38:42 2021

@author: hunterstuckey


"""

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y,u_z):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param u_z: accelaration in z-direction
        """

        # Define sampling time
        self.dt = dt

         # Define the  control input variables
        self.u = np.matrix([[u_x],[u_y],[u_z]])

        # Intial State
        self.x = np.matrix([[0],[0],[0],[1],[1],[1],[u_x],[u_y],[u_z]])

        # Define the State Transition Matrix A
        v = self.dt
        a = 0.5*self.dt**2
        self.A = np.matrix([[1, 0, 0, v, 0, 0, a, 0, 0],
                [0, 1, 0, 0, v, 0, 0, a, 0],
                [0, 0, 1, 0, 0, v, 0, 0, a],
                [0, 0, 0, 1, 0, 0, v, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, v, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, v],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0]])

        # Initial Covariance Matrix
        self.P = np.matrix([[100, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 100, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 100, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 100, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 100, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 100, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 100, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 100, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 100]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[a],[a],[a],[v],[v],[v],[1],[1],[1]])

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[1,0,0],
                           [0,1,0],
                           [0,0,1]])


    def predict(self):

        # Update time state
        self.x = np.dot(self.A, self.x)

        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:3]

    def update(self, z):

        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P 
        return self.x[0:3]


