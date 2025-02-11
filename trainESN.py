# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:01:56 2022

@author: Leon Scharwächter
"""

from echoStateNetwork import ESN
from utils import scaleMatrix, sparseMaker, getEchoStateProperty
import numpy as np

def initialize_ESN(n_input = 1, n_reservoir = 40, n_output = 1, scalingfactor = 1e-8):
    Win = (np.random.rand(n_input, n_reservoir)-0.5)
    Wr = (np.random.rand(n_reservoir, n_reservoir)-0.5)
    Wout = (np.random.rand(n_reservoir, n_output)-0.5)
    Wfb = (np.random.rand(n_output, n_reservoir)-0.5)
    
    # Scale matrices
    #Win = scaleMatrix(Win, scalingfactor)
    Wr = scaleMatrix(Wr, scalingfactor)
    #Wout = scaleMatrix(Wout, scalingfactor)
    #Wfb = scaleMatrix(Win, scalingfactor)

    # Create sparse matrices
    Win = sparseMaker(Win, 0.4)
    Wout = sparseMaker(Wout, 0.4)
    Wfb = sparseMaker(Wfb, 0.4)
    Wr = sparseMaker(Wr, 0.4)

    # transform matrix Wr to fulfil the echo state property
    Wr = getEchoStateProperty(Wr, 0.9)
    
    ESN = ESN(
        Win, Wr, Wout, Wfb,
        n_input=n_input, 
        n_reservoir=n_reservoir, 
        n_output=n_output
        )
return ESN

def create_signal(T1 = 5, T2 = 2, A1 = 0.9, A2 = 0.4, timesteps = 1000):
    t = np.arange(0, timesteps, 1)
    signal = A1*np.sin(t/T1)+A2*np.sin(t/T2)
    return signal[np.newaxis,:]

if __name__ == "__main__":
    ESN = initialize_ESN()
    signal = create_signal()
    T, Wout_new = ESN.train(signal, 700)
    prediction, NRMSE = ESN.predict(signal, T, Wout_new, 700, 300)
    print('NRMSE:',NRMSE)

