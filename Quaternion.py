"""
Class for Quaternion algebra
author: Nora Konrad
date: 07.05.2022

unit quaternion defined as q = q_w + q_x*i + q_y*j + q_z*k ∈ H, 
where {q_w, q_x, q_y, q_z } ∈ R, 
and {i, j, k} are three imaginary unit numbers deﬁned so that:
i^2 = j^2 = k^2 = ijk = -1,
and q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1.

paper reference: Quaternion Kinematics for Error-State Kalman Filter  
by Joan Solà, 2017

"""

import numpy as np 

class Quaternion:
    
    def __init__(self, value):

        """
        Define a Quaternion.
        
        Parameter
        ----------
        value : 1D array with 4 elements, e.g. [1, 0, 0, 0].

        Returns
        ----------
        out : Quaternion
            A Quaternion object, defined as q = q_w + q_x*i + q_y*j + q_z*k ∈ H,
            where {q_w, q_x, q_y, q_z } ∈ R, 
            and {i, j, k} are three imaginary unit numbers deﬁned so that:
            i^2 = j^2 = k^2 = ijk = -1,
            and q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1.
        """

        self.value = np.array(value)
    
    
    def magnitude(self):
        """
        Calculates the Quaternion's magnitude.
        """

        return np.sqrt(np.sum(np.power(self.value,2)))

    def normalize(self):
        """ 
        Normalizes the Quaternion, so that its magnitude equals 1.
        """

        n = self.magnitude()
        if n > 0:
            self.value = self.value/n


    def inv(self):
        """ 
        Computes the Quaternion's inverse (q = q_w - q_x*i - q_y*j - q_z*k).
        """

        q = self.value

        return self.__class__(np.array([q[0], -q[1], -q[2], -q[3]]))

    def quatprod(self, q2):
        """ 
        Computes the Quaternion product of two Quaternions.
        """

        q1 = self.value
        q2 = q2.value

        # Product (Joan Sola, eq (12))
        prod = np.array([q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                         q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                         q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                         q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0],
                         ])
        return self.__class__(prod)
    
    def rotation_matrix(self):
        """ 
        Computes the direction cosine matrix from the quaternion representing the orientation.
        """
        
        q = self.value
        
        # Rotation Matrix (Joan Sola, eq. (115))
        rot_matrix  = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                              [2*q[1]*q[2]+2*q[0]*q[3], q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*q[2]*q[3]-2*q[0]*q[1]],
                              [2*q[1]*q[3]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], q[0]**2-q[1]**2-q[2]**2+q[3]**2],
                              ])
        return rot_matrix

    