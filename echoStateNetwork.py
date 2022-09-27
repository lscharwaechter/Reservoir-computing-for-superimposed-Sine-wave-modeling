# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:39:30 2021

@author: Leon Scharwächter
"""
import matplotlib.pyplot as plt
import numpy as np

class ESN():
    def __init__(self, Win, Wr, Wout, Wfb,
                 n_input=1, n_reservoir=40, n_output=1):
        super().__init__()
        
        self.Win = Win
        self.Wr = Wr 
        self.Wout = Wout
        self.Wfb = Wfb 
        
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        
    def resetStates(self, steps):
        # Initialize net input and activation of
        # the reservoir and output layer with 0
        self.net_h = np.zeros((self.n_reservoir,steps))
        self.act_h = np.zeros((self.n_reservoir,steps))   
        self.net_hk = np.zeros((self.n_output,steps))
        self.act_hk = np.zeros((self.n_output,steps))
        
        
    def train(self, x, steps):
        self.resetStates(x.shape[1])
        
        prevt = 0
        for t in range(steps):           
            # Calculate activation of the reservoir:
            # First integrate feed-forward input
            for i in range(self.n_input):
                for j in range(self.n_reservoir):
                    self.net_h[j,t] += x[i,t]*self.Win[i,j]  
                    
            # Then integrate recurrent input with previous timestep
            for i in range(self.n_reservoir):
                for j in range(self.n_reservoir):
                    self.net_h[j,t] += self.act_h[j,prevt]*self.Wr[i,j]
            self.act_h = np.tanh(self.net_h) 
            
            # Calculate activation of the output layer
            for i in range(self.n_reservoir):
                for j in range(self.n_output):
                    self.net_hk[j,t] += self.act_h[i,t]*self.Wout[i,j]
            # No tanh nonlinearity
            self.act_hk = self.net_hk
              
            # Set previous timestep for next iteration
            if t > 0:
                prevt = t
                
        T = np.transpose(self.act_hk)
        
        # Teacher Forcing:
        # Replace the predicted output with 
        # the ground truth target (sequence)
        T = np.transpose(x)[:steps]
        
        # Determine M and new Wout
        M = np.transpose(self.act_h[:,:steps])
        Wout_new = np.dot(np.linalg.pinv(M),T)
        
        plt.plot(T[:,0])
        
        return T, Wout_new
    
    def predict(self, x, T, Wout_new, trainingSteps, testingSteps):
        
        # Start the Closed Loop with the last output of
        # the training procedure
        cl_output = T[-1,:]
        
        prevt = trainingSteps
        for t in range(testingSteps):
            
            t = t+trainingSteps
            
            # Calculate activation of the reservoir,
            # this time using Wfb
            for i in range(self.n_input):
                for j in range(self.n_reservoir):
                    self.net_h[j,t] += cl_output*self.Wfb[i,j]  
            for i in range(self.n_reservoir):
                for j in range(self.n_reservoir):
                    self.net_h[j,t] += self.act_h[j,prevt]*self.Wr[i,j]
            self.act_h = np.tanh(self.net_h) 
            
            # Calculate activation of the output layer,
            # this time using Wout_new
            for i in range(self.n_reservoir):
                for j in range(self.n_output):
                    self.net_hk[j,t] += self.act_h[i,t]*Wout_new[i,j]
            # No tanh nonlinearity
            self.act_hk = self.net_hk
            
            cl_output = self.act_hk[:,t]
              
            # Set previous timestep for next iteration
            prevt = t
        
        # Calculate Normalized RMSE    
        NRMSE = np.sqrt(np.mean((self.act_hk[:,trainingSteps:]-x[:,trainingSteps:])**2)/np.var(x[:,trainingSteps:]))      
        
        ### PLOT ###
        x_values = np.arange(trainingSteps,trainingSteps+testingSteps)
        x_values = np.transpose(x_values)
        plt.plot(x_values,self.act_hk[0,trainingSteps:])
        #plt.vlines(trainingSteps, ymin = -1, ymax = 1, colors='r')
        
        return self.act_hk, NRMSE
 
