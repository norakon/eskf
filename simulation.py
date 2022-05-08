import numpy as np
from Quaternion import Quaternion
from ESKF import ESKF
import math
import scipy.io 
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # load pre-simulated IMU and Vive Tracker measurements from a .mat - file
    mat = scipy.io.loadmat('simulated_measurement/Circle_2020-11-30.mat')

    imu_time = mat['IMU']['time'][0][0]
    imu_time = imu_time.reshape(imu_time.shape[1])

    # The IMU measurements were simulated with MATLAB Navigation Toolbox, with:
    # - Accelerometer.BiasInstability = 9.0e-4 rm/s², 
    # - Accelerometer.RandomWalk = 3.5e-4 (m/s²)*sqrt(Hz), 
    # - Gyroscope.BiasInstability = 1.5e-4 rad/s and 
    # - Gyroscope.RandomWalk = 2.0e-4  (rad/s)*sqrt(Hz).
    imu_linacc = mat['IMU']['LinAcc'][0][0]
    imu_angvel = mat['IMU']['AngVel'][0][0]

    # The Vive measurements were simulated as: 
    # - additive white Gaussian noise, with sigma = 0.016 m.
    vive_time = mat['tracker']['time'][0][0]
    vive_time = vive_time.reshape(vive_time.shape[1])

    vive_pos = mat['tracker']['Pos'][0][0]
    vive_vel = mat['tracker']['Vel'][0][0]
    vive_q = mat['tracker']['Orient'][0][0]

    # for plots
    time = np.zeros(30000)
    pos = np.zeros((30000,3))
    err = np.zeros((30000,3))
    cov = np.zeros((30000,15))

    # Initiate figure
    fig = plt.figure()
    fig.set_size_inches(14.5, 11.5)

    ax1 = fig.add_subplot(1, 2, 1,projection="3d")
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 4)
    ax4 = fig.add_subplot(3, 2, 6)

    ax1.set_xlabel('x / m')
    ax1.set_ylabel('y / m')
    ax1.set_zlabel('z / m')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_zlim(0, 15)

    ax2.set_xlabel('t / s')
    ax2.set_ylabel('err_x / m')

    ax3.set_xlabel('t / s')
    ax3.set_ylabel('err_y / m')

    ax4.set_xlabel('t / s')
    ax4.set_ylabel('err_z / m')

    # INIT ESKF
    nom_p = np.array([5.,0.,0.])
    nom_v = vive_vel[:,0]
    nom_q = Quaternion(vive_q[:,0])
    nom_ab = np.zeros((3))
    nom_wb = np.zeros((3))
    
    imu_sigma = np.array([0.5,    # m/s^2
                          0.2,    # rad/s
                          0.001,  # m/s^2*sqrt(s)
                          0.001]) # rad/s*sqrt(s)

    vive_sigma = np.array([0.01,    # m
                           0.05,    # m/s
                           0.01])

    cov_factor = np.concatenate((0.1 * np.ones((3,1)),    # m
                                 0.1 * np.ones((3,1)),    # m/s
                                 0.01 * np.ones((3,1)),   # rad
                                 0.001 * np.ones((3,1)),  # m/s^2
                                 0.002 * np.ones((3,1))), # rad/s
                                 axis=0)
    cov_matrix = cov_factor * np.ones(15)  

    ACC_GRAVITY = -9.81 # m/s^2

    # initiate Error State Kalman Filter object
    eskf = ESKF(nom_p=nom_p, nom_v=nom_v, nom_q=nom_q, nom_ab=nom_ab, nom_wb=nom_wb, imu_sigma=imu_sigma, 
                vive_sigma=vive_sigma, cov_matrix=cov_matrix, ACC_GRAVITY=ACC_GRAVITY)

    # initialize the while loop
    doLoop = True
    iu = 1
    iv = 1
    i = 0
    t = np.zeros((imu_time.size+vive_time.size))
    t[1] = vive_time[1]

    # Simulate, as if the different measurements came asynchronously in real-time
    while doLoop:
    
        # break condition
        if iu == (imu_time.size-1) or iv == (vive_time.size-1) or i >= time.size-1:
            doLoop = False
        
        if doLoop == False: 
            break 

        i = i + 1; 
        
        ## find current time step
        # find current imu time
        while 1:
            if imu_time[iu-1] <= t[i-1] and imu_time[iu] >= t[i-1]:
                break
            else:
                iu = iu + 1
            
        # find current vive time
        while 1:
            if vive_time[iv-1] <= t[i-1] and vive_time[iv] >= t[i-1]:
                break
            else:
                iv = iv + 1
        
        # check which sensor provided the measurement for the current time step
        t[i] = t[i-1]
        while t[i] == t[i-1]:
            if imu_time[iu] <= vive_time[iv]:
                t[i] = imu_time[iu]
                iu = iu + 1
                curr_sensor = 'IMU'
            else:
                t[i] = vive_time[iv]
                iv = iv + 1
                curr_sensor = 'Vive'
            
        # current measurement vector u (6x1 vector) and output vector y (10x1 vector)
        u = np.block([imu_linacc[:,iu], imu_angvel[:,iu]])
        y = np.block([vive_pos[:,iv], vive_vel[:,iv],vive_q[:,iv]]) 
        
        ## ESKF:

        # delta t between two consecutive measurements
        dt = t[i] - t[i-1]
        
        # predict a priori state
        eskf.prediction(imu_meas=u,dt=dt)
        
        # compute a posteriori state
        if curr_sensor == 'Vive':
            eskf.correction(vive_meas=y)

        # second index to save only every 50th measurement for plots       
        j = i // 100
        
        # Plot x,y,z trajectory
        if (i % 100 == 0):  # don't plot every iteration
            
            # collect data for plots
            time[j] = t[i]
            pos[j,:] = eskf.nom_p.transpose()
            err[j,:] = eskf.err_p.transpose()
            cov[j,:] = np.sqrt(eskf.cov_matrix.diagonal())

            # plot
            ax1.plot(pos[j,0], pos[j,1], pos[j,2],'.g', label='3D Position')
            ax1.grid(True)

            ax2.plot(time[j], err[j,0],'.b',label='error state p_x')
            ax2.plot(time[j], -cov[j,0], '.r',label='stddev boundaries')
            ax2.plot(time[j], +cov[j,0], '.r')
            ax2.grid(True)

            ax3.plot(time[j],  err[j,1],'.b',label='error state p_y')
            ax3.plot(time[j], -cov[j,1], '.r',label='stddev boundaries')
            ax3.plot(time[j], +cov[j,1], '.r')
            ax3.grid(True)

            ax4.plot(time[j], err[j,2],'.b',label='error state p_z')
            ax4.plot(time[j], -cov[j,2], '.r',label='stddev boundaries')
            ax4.plot(time[j], +cov[j,2], '.r')
            ax4.grid(True)

            plt.pause(0.001)

            # plot the legends only once
            if j < 2:
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax4.legend()

            

