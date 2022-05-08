# ESKF localization
<img src = "ESKF-localization-simulation.gif" width=640>

Sensor fusion is the process of merging different sensor measurements to obtain enhanced
information with less uncertainty. This repo, implements an Error State Kalman Filter that is
able to process measurements of a MEMS IMU and Vive Tracker to improve the pose estimation 
of a fully actuated hexacopter. The ESKF is based on Joan Solà's paper [1].

## ESKF vs EKF

The advantage of the ESKF compared to the standard implementation of EKF is that merely
the errors of the modelled system are estimated and corrected. These errors have signiﬁ-
cantly lower temporal dynamics than the movements of the vehicle itself, which means that
the dynamics of the ﬁlter are largely uncoupled from the system dynamics. [2, chap. 4.3]
Due to the expected values of the errors being zero, the error state is always small. That 
means that all second order products and hence the linearisation errors are negligible. [1]
This leads to a higher accuracy than an EKF with full states. Besides, there are no parameter
singularities or gimbal lock problems because the error state system always operates close
to the origin. [1] Additionally, the ESKF is not only able to estimate the errors of position,
velocity and orientation but also to estimate the biases of the accelerometer and gyroscope.
This is the key to keeping the estimated position, velocity and attitude errors small.

## Requirements

- python=3.9
- pip
- scipy
- numpy
- matplotlib

### How to install the requirements

1. Clone this repository.

> git clone 


2. Install the required libraries.

using conda :

> conda env create -f requirements/environment.yml
 
using pip :

> pip install -r requirements/requirements.txt

## How to start the simulation

1. Activate Conda Environment :

> conda activate ESKF_sim

2. Execute python script from the directory :

> python simulation.py

## References
- [1] J. Solà, “Quaternion kinematics for the error-state Kalman ﬁlter”, 2017. arXiv: [1711.02508](https://arxiv.org/abs/1711.02508).
- [2] N. Steinhardt, “Eine Architektur zur Schätzung kinematischer Fahrzeuggrößen mit integrierter Qualitätsbewertung durch Sensordatenfusion”, PHD Thesis, Technische Universität Darmstadt, 2014, ISBN: 9783183781126.