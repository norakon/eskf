"""
Error State Kalman Filter
author: Nora Konrad
date: 07.05.2022

paper reference: Quaternion Kinematics for Error-State Kalman Filter   
by Joan Solà
"""

import numpy as np
import math
from Quaternion import Quaternion

class ESKF:
    
    def __init__(self, nom_p, nom_v, nom_q, nom_ab, nom_wb, imu_sigma, vive_sigma, cov_matrix, ACC_GRAVITY):
        """
        Create an Error State Kalman Filter object.
        
        Parameters
        ----------
        nom_p : (3x1) position p in m
        nom_v : (3x1) velocity v in m/s 
        nom_q : (4x1) Orientation q 
        nom_ab : (3x1) Linear Acceleration bias in m/s² 
        nom_wb : (3x1) Angular Velocity bias in rad/s 

        imu_sigma : (4x1) IMU standard deviations of: 
                    - acc measurement noise, 
                    - gyr measurement noise, 
                    - acc perturbation, 
                    - gyr perturbation.

        vive_sigma :  (3x1) Vive Tracker standard deviations of: 
                    - position measurement noise, 
                    - velocity measurement noise, 
                    - Quaternion measurement noise. 

        cov_matrix : (15x15) covariance matrix P of error state ex

        ACC_GRAVITY : (1x1) gravitational acceleration
 
        Returns
        ----------
        out : ESKF
            An Error State Kalman Filter object.
        """

        # Nominal State
        self.nom_p = nom_p
        self.nom_v = nom_v
        self.nom_q = nom_q
        self.nom_ab = nom_ab
        self.nom_wb = nom_wb

        # Error State
        self.err_p = np.zeros((3))
        self.err_v = np.zeros((3))
        self.err_q = np.zeros((3))
        self.err_ab= np.zeros((3))
        self.err_wb = np.zeros((3))

        # Stddev IMU measurement
        self.imu_sigma = imu_sigma

        # Stddev vive measurement
        self.vive_sigma = vive_sigma

        # Covariance matrix P, corresponding to error state
        self.cov_matrix = cov_matrix

        # Measurement Matrix H
        self.H = np.zeros((10,15))

        # Vive Tracker covariance matrix
        self.V = np.ones((10,10))

        # Transition matrix
        self.Fx = np.zeros((16,16))

        # Control matrix
        self.Fu = np.zeros((15,6))

        # Perturbation matrix
        self.Fw = np.zeros((15,6))

        # IMU covariance matrices
        self.U = np.zeros((6,6))
        self.W = np.zeros((6,6))

        # Direction Cosine matrix
        self.Rot_q = np.zeros((3,3))

        # Gravitational acceleration        
        self.ACC_GRAVITY = ACC_GRAVITY


    def skewm(self,vec):    
        """
        Computes Skew-symmetric matrix, to compute Vector product.

        Parameter
        ----------
        vec : (3x1) Vector.

        Returns
        ----------
        (3x3) skew-symmetric matrix
        """

        vec = vec.flatten()
        skew = np.array([[0, -vec[2],  vec[1]],
                [vec[2],  0,  -vec[0]],
                [-vec[1],  vec[0],   0]])
        return skew

    def calc_Fx(self, imu_meas, dt):  
        """
        Computes Transition Matrix for Error State ex.

        Parameters
        ----------
        imu_meas : (6x1) IMU measurement:
                    - (3x1) linear acceleration in m/s^2
                    - (3x1) angular velocity in rad/s
        dt : (1x1) delta t is the time between two measurements
        """

        # save the measurement in separate variables
        imu_am = imu_meas[0:3]
        imu_wm = imu_meas[3:6]

        # calc Rotation matrix
        self.Rot_q = self.nom_q.rotation_matrix()

        # subtract bias
        w_diff = imu_wm - self.err_wb
        w_norm = np.sqrt(np.sum(np.power(w_diff,2)))
        
        # using Rodrigues rotation formula for integration of angular error 
        # (Joan Sola, eq.(347))
        if w_norm == 0:
            uRod = w_diff
        else:
            uRod = w_diff/w_norm
        phiRod = np.sqrt(np.sum(np.power(w_diff*dt,2)))
        RodrTrans = np.eye(3) - self.skewm(uRod)*math.sin(phiRod) + (self.skewm(uRod))**2 * (1 - math.cos(phiRod)); 
        
        # update Transition matrix
        self.Fx = np.block([[np.eye(3), np.eye(3)*dt, -0.5*self.Rot_q*self.skewm(imu_am-self.nom_ab)*dt**2, -0.5*self.Rot_q*dt**2, 1/6*self.Rot_q*dt**3],
                   [np.zeros((3,3)), np.eye(3), -(self.Rot_q)*self.skewm(imu_am - self.nom_ab)*dt, -(self.Rot_q)*dt, 0.5*self.Rot_q*self.skewm(imu_am-self.nom_ab)*dt**2],
                   [np.zeros((3,3)), np.zeros((3,3)), RodrTrans, np.zeros((3,3)), -np.eye(3)*dt],
                   [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
                   [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]])
        

    def calc_Fu(self, dt): 
        """
        Computes the Control Matrix.

        Parameter
        ----------
        dt : (1x1) delta t is the time between two measurements.
        """

        B = np.block([[np.zeros((3,3)), np.zeros((3,3))],
             [-(self.Rot_q), np.zeros((3,3))],
             [np.zeros((3,3)), np.eye(3)],
             [np.zeros((3,3)), np.zeros((3,3))],
             [np.zeros((3,3)), np.zeros((3,3))]])

        # update Control matrix
        self.Fu = B*dt
    
    def calc_Fw(self): 
        """
        Computes the Perturbation Matrix.
        """

        self.Fw = np.block([[np.zeros((3,3)), np.zeros((3,3))],
                   [np.zeros((3,3)), np.zeros((3,3))],
                   [np.zeros((3,3)), np.zeros((3,3))],
                   [np.eye(3), np.zeros((3,3))],
                   [np.zeros((3,3)), np.eye(3)]])
    
    def calc_U_W(self, dt): 
        """
        Computes the Noise and Perturbation Covariance Matrices.

        Parameter
        ----------
        dt : (1x1) delta t is the time between two measurements.
        """

        # (Joan Sola, eq. (452))
        Uc = np.block([[self.imu_sigma[0]**2*np.eye(3), np.zeros((3,3))],
              [np.zeros((3,3)), self.imu_sigma[1]**2*np.eye(3)]])
        self.U = Uc
        
        Wc = np.block([[self.imu_sigma[2]**2*np.eye(3), np.zeros((3,3))],
              [np.zeros((3,3)), self.imu_sigma[3]**2*np.eye(3)]])
        self.W = Wc*dt

    def strapdown(self, imu_meas, dt):
        """
        Computes the Nominal State, based on IMU measurements.

        Parameters
        ----------
        imu_meas : (6x1) IMU measurement:
                    - (3x1) linear acceleration in m/s^2
                    - (3x1) angular velocity in rad/s
        dt : (1x1) delta t is the time between two measurements
        """

        # save the measurement in separate variables
        imu_am = imu_meas[0:3]
        imu_wm = imu_meas[3:6]

        # calculate orientation
        w_diff = imu_wm + self.nom_wb
        w_diff_norm = np.linalg.norm(w_diff,2)

        if (w_diff_norm == 0):
            q_wd = Quaternion([1,0,0,0])
        else:
            q_wd = Quaternion([math.cos(0.5*w_diff_norm*dt),
                            (w_diff[0]/w_diff_norm * math.sin(0.5*w_diff_norm*dt)),
                            (w_diff[1]/w_diff_norm * math.sin(0.5*w_diff_norm*dt)),
                            (w_diff[2]/w_diff_norm * math.sin(0.5*w_diff_norm*dt))])
            self.nom_q = Quaternion.quatprod(self.nom_q,q_wd)
            self.nom_q.normalize()

        # compensate gravity before integration
        acc_est = np.dot(self.nom_q.rotation_matrix(),(imu_am-self.nom_ab)) - np.array([0, 0, self.ACC_GRAVITY])

        # velocity integration
        self.nom_v = self.nom_v + (acc_est)*dt

        # Position integration
        self.nom_p = self.nom_p + self.nom_v*dt + (acc_est)*dt**2

        # bias 
        self.nom_ab = self.nom_ab
        self.nom_wb = self.nom_wb

        
    def calc_H_W(self):
        """
        Computes the Measurement Matrix and Vive Tracker covariance matrix.
        """

        # this H Matrix only takes the Trackers Position, Velocity and
        # Orientation
        Hx = np.concatenate((np.eye(10), np.zeros((10,6))), axis=1)

        # compute the Jacobian of the true state with respect to the error state.
        Qetheta = 0.5*np.block([[np.zeros((1,3))],
                                [np.eye(3)]])
                    
        Xex = np.block([[np.eye(6), np.zeros((6,9))],
                        [np.zeros((4,6)), Qetheta, np.zeros((4,6))],
                        [np.zeros((6,9)), np.eye(6)]])

    
        # (Joan Sola, equation (277))
        self.H = Hx @ Xex
        
        self.V = np.block([[self.vive_sigma[0]**2*np.eye(3), np.zeros((3,7))],
                      [np.zeros((3,3)), self.vive_sigma[1]**2*np.eye(3), np.zeros((3,4))],
                      [np.zeros((4,6)), self.vive_sigma[2]**2*np.eye(4)]])


    def nomstate_update(self):
        """
        Updates the Nominal State, involving the error state.
        """

        self.nom_p = self.nom_p + self.err_p
        self.nom_v = self.nom_v + self.err_v
        
        dtheta_norm = np.sqrt(np.sum(np.power(self.err_q,2)))
        
        if dtheta_norm == 0:
                q_Sola = Quaternion([1,0,0,0])
        else:
                q_Sola = Quaternion([math.cos(0.5*dtheta_norm),
                                     self.err_q[0]/dtheta_norm * math.sin(0.5*dtheta_norm),
                                     self.err_q[1]/dtheta_norm * math.sin(0.5*dtheta_norm),
                                     self.err_q[2]/dtheta_norm * math.sin(0.5*dtheta_norm)])
        
        self.nom_q = Quaternion.quatprod(self.nom_q,q_Sola.inv())
        self.nom_q.normalize()

    
        self.nom_ab = self.nom_ab + self.err_ab
        self.nom_wb = self.nom_wb + self.err_wb

    def eskf_reset(self):
        """
        Resets the error state and covariance matrix back to zero.
        """

        self.err_p = np.zeros(3)
        self.err_v = np.zeros(3)
        self.err_q = np.zeros(3)
        self.err_ab = np.zeros(3)
        self.err_wb = np.zeros(3)

        G = np.eye(15)

        self.cov_matrix = G*self.cov_matrix*G.transpose()
        

    def prediction(self, imu_meas, dt):
        """
        Computes the a priori error state and its covariance matrix and propagates the nominal state in time.

        Parameters
        ----------
        imu_meas : (6x1) IMU measurement:
                    - (3x1) linear acceleration in m/s^2
                    - (3x1) angular velocity in rad/s
        dt : (1x1) delta t is the time between two measurements
        """

        self.calc_Fx(imu_meas=imu_meas, dt=dt)

        self.calc_Fu(dt=dt)

        self.calc_Fw()

        self.calc_U_W(dt=dt)

        self.strapdown(imu_meas=imu_meas, dt=dt)

        self.cov_matrix = self.Fx @ self.cov_matrix @ self.Fx.transpose() + self.Fu @ self.U @ self.Fu.transpose() + self.Fw @ self.W @ self.Fw.transpose()

        


    def correction(self,vive_meas):
        """
        Computes the improved a posteriori error state and covariance matrix and updates the nominal state.

        Parameter
        ----------
        vive_meas : (10x1) Vive Tracker measurement:
                    - (3x1) position measurement in m
                    - (3x1) velocity in m/s
                    - (4x1) quaternion
        """

        # save the vive measurement in separate variables
        vive_p = vive_meas[0:3]
        vive_v = vive_meas[3:6]
        vive_q = Quaternion(vive_meas[6:10])

        # calc H and W matrices
        self.calc_H_W()

        # Kalman Gain
        K = self.cov_matrix @ self.H.transpose() @ np.linalg.inv(self.H @ self.cov_matrix @ self.H.transpose() + self.V)

        # update covariance matrix: Joseph Form
        self.cov_matrix = (np.eye(15) - K @ self.H) @ self.cov_matrix @ np.transpose(np.eye(15) - K @ self.H) + K @ self.V @ K.transpose() 
        
        # update error state 
        eq = Quaternion.quatprod(vive_q.inv(),self.nom_q)   # delta_q = q_v0 x q_i0*
        
        ex = K @ np.concatenate((vive_p-self.nom_p, vive_v-self.nom_v,eq.value), axis=0)

        self.err_p = ex[0:3]
        self.err_v = ex[3:6]
        self.err_q = ex[6:9]
       
        self.nomstate_update() 

        self.eskf_reset()