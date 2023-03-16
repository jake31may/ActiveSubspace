#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:37:19 2023

@author: jakehollins
"""
# Libraries/Modules Required:
import numpy as np
from pyDOE import lhs
import matlab.engine
from kneed import KneeLocator


class AS_model:
    '''Builds an active subspace from an initial sample,'''
    def __init__(self,dims=11,nModes=10, CovarianceMatrix = None):
        ''''Initalises Active subspace model'''

        self.CovList = list([AS_sum,AS_product,AS_transform]) # List of available multi-output covariance matrices
        
        if all([isinstance(dims, int), isinstance(nModes, int)]) == True: # Checks if inputs are integers
           self.dims = dims
           self.nModes = nModes
        else:
            raise TypeError('Dimensions and number of modes must be an integer')
        
            
        if CovarianceMatrix is None:
            self.Cov = AS_sum
        
        #if self.CovList.count(CovarianceMatrix)== 1: # Checks if Cov matrix setting is entered correctly
        #    self.Cov = CovarianceMatrix
        #else:
        #    raise TypeError('Covariance matrix does not exist. Use any of the following: \n AS_sum, AS_product, AS_transform')
        
        self.eng = matlab.engine.start_matlab()
        
        # Inserts default settings:
        self.AS_locator = 'knee'        # active variables locator default: 'knee' for kneedling algo.
        
        
    #-------------------------------------------------------------------------
    # Functions that edit settings: 
    def Cov_Matrix(self, CovarianceMatrix):
        '''Sets/changes covariance matrix'''
        if self.CovList.count(CovarianceMatrix)== 1:
            self.Cov = CovarianceMatrix
        else:
            raise TypeError('Covariance matrix does not exist. Use any of the following: \n AS_sum, AS_product, AS_transform')
        
        
    def Num_Modes(self, nModes):
        '''Changes number of modes for output'''
        if isinstance(nModes, int) == True:
           self.nModes = nModes
        else:
            raise TypeError('Number of modes must be an integer')
            
            
    def Num_Dims(self, dims):
        '''Changes number of dimensions for input'''
        if isinstance(dims, int) == True:
           self.dims = dims
        else:
            raise TypeError('Number of modes must be an integer')
    
    
    #-------------------------------------------------------------------------
    # Intialising model and adding samples
    def init_sample(self,samples,sample_range=[0,1],sampler='lhs'):
        ''' Initialises n samples within a given range'''
        
        self.range = sample_range
        
        print('Generating sample points...')
        if sampler == 'lhs':
            self.samples = sample_range[0] + (sample_range[1]-sample_range[0])*lhs(self.dims,samples)
        
        # Could add other samplers here...
        #elif sampler == 'sobol':
        #elif sampler == 'cma-es':
            
        
        print('Running model...')
        Output = self.eng.Wingbox_eval_AS(matlab.double(self.samples),self.nModes)
        Output = np.asarray(Output)
        
        self.Frequencies = Output[:,:self.nModes]
        self.Gradients = Output[:,self.nModes:]
                
        
    def add_samples(self,n_samples,sample_range = None, sampler = 'lhs'):
        '''Add additional samples to the model'''
        
        if sample_range is not None:
            self.sample_range = sample_range
        
        print('Generating new sample points...')
        if sampler == 'lhs':
            samples = self.range[0] + (self.range[1]-self.range[0])*lhs(self.dims,n_samples)
        
        # Could add other samplers here...
        #elif sampler == 'sobol':
        #elif sampler == 'cma-es':
            
        print('Running model...')
        Output = self.eng.Wingbox_eval_AS(matlab.double(samples),self.nModes)
        Output = np.asarray(Output)
        Frequencies = Output[:,:self.nModes]
        Gradients = Output[:,self.nModes:]
        
        self.Frequencies = np.append(self.Frequencies,Frequencies,axis=0)
        self.Gradients = np.append(self.Gradients,Gradients,axis=0)
    
    
    def FindSubspace(self,AS_locator = None):
        ''' Determine active subspace by selecting active variables. Multiple methods available.'''
        
        if AS_locator is not None:
            self.AS_locator = AS_locator    # Sets locator (default: 'knee')

        
        W,V = self.Cov(Gradients=self.Gradients,nModes=self.nModes,dims=self.dims)
        
        if self.AS_locator == 'knee':
            k = KneeLocator(range(self.dims),W,direction='decreasing',curve='convex').knee
        
        elif self.AS_locator == 'tolerance':
            self.tol = int(input ("What is the tolerance for active variables? "))
            k = AS_LocTol(Eigenvalues = W,tolerance = self.tol)
        else:
            raise TypeError('Active subspace locator not found')
            return
        
        return W[:k],V[:,:k]
            
    
#----------------------------------------------------------------------------
# Multi-output covariance functions
def AS_sum(Gradients,nModes,dims):
    """ Produces multi-output covariance matrix by summing"""
    nSamples = np.shape(Gradients)[0]
    C = np.zeros([dims,dims])
    for i in range(nSamples):
        c = np.reshape(Gradients[i,:],[nModes,dims])
        C = (c.T @ c) + C
    C /= nSamples
    
    w, v = np.linalg.eig(C)
    
    w = np.real(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    return w,np.abs(v)

def AS_product(self):
    """" Produces multi-output covariance matrix by vectorising gradients """
    nSamples = np.shape(self.Gradients)[0]
    C = np.zeros([self.dims**2,self.dims**2])
    for i in range(nSamples):
       c = self.Gradients[i,:]
       C = (c.T @ c) + C
   
    C /= nSamples
   
    w, v = np.linalg.eig(C)
    
    w = np.real(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    
    return w,np.abs(v)

def AS_transform(self):
    """" Produces multi-output covariance matrix by vectorising gradients,
    then transforming"""
    nSamples = np.shape(self.Gradients)[0]
    C = np.zeros([self.dims**2,self.dims**2])
    for i in range(nSamples):
       c = self.Gradients[i,:]
       C = (c.T @ c) + C
    C /= nSamples
   
    T = np.tile(np.eye(self.dims), self.dims)
    C = (T @ C) @ T.T
   
    w, v = np.linalg.eig(C)
    
    w = np.real(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    
    return w,np.abs(v)

#-----------------------------------------------------------------------------
# Other Functions
def MAC_Error(V1,V2):
    """" Calculates modal assurance criterion (MAC) error between two matrices"""
    nom = np.abs(np.transpose(V1) @ V2)**2
    denom1 = np.diag(np.transpose(V1) @ V1)
    denom2 = np.diag(np.transpose(V2) @ V2)
    denom = denom1 @ denom2
    
    return np.sum(np.diag(nom/denom))

def AS_LocTol(Eigenvalues,tolerance):
    ''' Locates active variables based on specified tolerance'''
    
    Eigenvalues /= Eigenvalues[0] #Normalises eigenvalues
    
    return np.argmin(Eigenvalues<tolerance)-1