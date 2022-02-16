# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:01:56 2022

@author: Leon Scharwächter
"""

from echoStateNetwork import ESN
from utils import scaleMatrix, sparseMaker, getEchoStateProperty
import numpy as np

# Model parameters
n_input = 1
n_reservoir = 40
n_output = 1

scalingfactor = 1e-8

Win = (np.random.rand(n_input, n_reservoir)-0.5)
Wr = (np.random.rand(n_reservoir, n_reservoir)-0.5)
Wout = (np.random.rand(n_reservoir, n_output)-0.5)
Wfb = (np.random.rand(n_output, n_reservoir)-0.5)

#Win = scaleMatrix(Win, scalingfactor)
Wr = scaleMatrix(Wr, scalingfactor)
#Wout = scaleMatrix(Wout, scalingfactor)
#Wfb = scaleMatrix(Win, scalingfactor)

Win = sparseMaker(Win, 0.4)
Wout = sparseMaker(Wout, 0.4)
Wfb = sparseMaker(Wfb, 0.4)
Wr = sparseMaker(Wr, 0.4)

Wr = getEchoStateProperty(Wr, 0.9)

ESN = ESN(
    Win, Wr, Wout, Wfb,
    n_input=n_input, 
    n_reservoir=n_reservoir, 
    n_output=n_output
    )

####### Create Signal #######

T1 = 5
T2 = 2
A1 = 0.9
A2 = 0.4
t = np.arange(0, 1000, 1)
signal = A1*np.sin(t/T1)+A2*np.sin(t/T2)
signal = signal[np.newaxis,:]

#############################

T, Wout_new = ESN.training(signal, 700)
prediction, NRMSE = ESN.testing(signal, T, Wout_new, 700, 300)

print('NRMSE:',NRMSE)

