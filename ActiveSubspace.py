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
import io
from scipy.stats import qmc
from scipy.optimize import minimize


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
            self.Cov = AS_single
        
        #if self.CovList.count(CovarianceMatrix)== 1: # Checks if Cov matrix setting is entered correctly
        #    self.Cov = CovarianceMatrix
        #else:
        #    raise TypeError('Covariance matrix does not exist. Use any of the following: \n AS_sum, AS_product, AS_transform')
        
        self.eng = matlab.engine.start_matlab()
        
        # Inserts default settings:
        self.AS_locator = 'knee'        # active variables locator default: 'knee' for kneedling algo.
        
        self.tolerance = 0.01        # active variables locator default: 'knee' for kneedling algo.
        
        
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
    def init_sample(self,samples=None,n_samples=5,sample_range=[0,1],sampler='lhs'):
        ''' Initialises n samples within a given range'''
        
        self.range = sample_range
        
        if samples is not None:
            self.samples = samples
        else:
            print('Generating sample points...')
            if sampler == 'lhs':
                self.samples = sample_range[0] + (sample_range[1]-sample_range[0])*lhs(self.dims,n_samples)
            
            elif sampler == 'MaxDistance':
                ones = np.ones([1,self.dims])
                self.samples = ShotgunMaxDis(Eval_points=ones,output_samples=n_samples,constraints=sample_range,shells=200)
            
            elif sampler == 'Halton':
                self.sampler = qmc.Halton(d=self.dims)
                samples = self.sampler.random(n=n_samples)
                self.samples = qmc.scale(samples,self.range[0],self.range[1])
            
            
            # Could add other samplers here...
            #elif sampler == 'sobol':
            #elif sampler == 'cma-es':
        
        
        print('Running model...')
        #Output = self.eng.Wingbox_eval_AS(matlab.double(self.samples),self.nModes)
        Output = self.eng.Wingbox_ROM(matlab.double(self.samples),self.nModes)
        Output = np.asarray(Output)
        
        self.Frequencies = Output[:,:self.nModes]
        self.Gradients = Output[:,self.nModes:]
                
        
    def add_samples(self,n_samples,sample_range = None, sampler = 'lhs'):
        '''Add additional samples to the model'''
        
        if sample_range is not None:
            self.range = sample_range
        
        print('Generating new sample points...')
        if sampler == 'lhs':
            samples = self.range[0] + (self.range[1]-self.range[0])*lhs(self.dims,n_samples)
            
        elif sampler == 'MaxDistance':
            samples = ShotgunMaxDis(Eval_points=self.samples,output_samples=n_samples,constraints=self.range,shells=200)
            
        elif sampler == 'Halton':
            samples = self.sampler.random(n=n_samples)
            samples = qmc.scale(samples,self.range[0],self.range[1])
        
        elif sampler == 'MaxDisOpt':
            samples = MaxDisScipy(Eval_points = self.samples,output_samples=n_samples,constraints=self.range)
            
        
        # Could add other samplers here...
        #elif sampler == 'sobol':
        #elif sampler == 'cma-es':
            
        print('Running model...')
        #Output = self.eng.Wingbox_eval_AS(matlab.double(samples),self.nModes)
        Output = self.eng.Wingbox_ROM(matlab.double(samples),self.nModes)
        Output = np.asarray(Output)
        Frequencies = Output[:,:self.nModes]
        Gradients = Output[:,self.nModes:]
        
        self.Frequencies = np.append(self.Frequencies,Frequencies,axis=0)
        self.Gradients = np.append(self.Gradients,Gradients,axis=0)
        self.samples = np.append(self.samples,samples,axis=0)
    
    
    def FindSubspace(self,Gradients=None,CovarianceMatrix= None,AS_locator = None,tolerance=None,single_mode=None):
        ''' Determine active subspace by selecting active variables. Multiple methods available.
        Returns eigenvalues W and eigenvectors V.'''
        
        if AS_locator is not None:
            self.AS_locator = AS_locator    # Sets locator (default: 'knee')
        
        if CovarianceMatrix is not None:
            self.Cov = CovarianceMatrix
            
        if single_mode is not None:
            self.mode = single_mode
            
        if tolerance is not None:
            self.tol = tolerance
        
        if Gradients is None:
            Gradients = self.Gradients
        
        Grad_norm = Gradients#/np.tile(np.mean(self.Frequencies,axis=0),self.dims)
        W,V = self.Cov(Gradients=Grad_norm,nModes=self.nModes,dims=self.dims,mode =self.mode)
        
        if self.AS_locator == 'knee':
            k = KneeLocator(range(self.dims),W,direction='decreasing',curve='convex').knee
        
        elif self.AS_locator == 'tolerance':
            self.tol = tolerance
            k = AS_LocTol(Eigenvalues = W,tolerance = self.tol)
        else:
            raise TypeError('Active subspace locator not found')
            return
        
        return W[:k],V[:,:k]
            
    
#----------------------------------------------------------------------------
# Multi-output covariance functions
def AS_sum(Gradients,nModes,dims,mode=None):
    """ Produces multi-output covariance matrix by summing"""
    nSamples = np.shape(Gradients)[0]
    C = np.zeros([dims,dims])
    for i in range(nSamples):
        c = np.reshape(Gradients[i,:],[nModes,dims])
        C += (c.T @ c)
    C /= nSamples
    
    w, v = np.linalg.eig(C)
    
    w = np.abs(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    return w,np.abs(v)

def AS_product(Gradients,nModes,dims,mode=None):
    """" Produces multi-output covariance matrix by vectorising gradients """
    nSamples = np.shape(Gradients)[0]
    C = np.zeros([dims**2,dims**2])
    for i in range(nSamples):
       c = np.reshape(Gradients[i,:],[1,-1])
       C += (c.T @ c)
   
    C /= nSamples
   
    w, v = np.linalg.eig(C)
    
    w = np.rabs(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    
    return w,np.abs(v)

def AS_transform(Gradients,nModes,dims,mode=None):
    """" Produces multi-output covariance matrix by vectorising gradients,
    then transforming"""
    nSamples = np.shape(Gradients)[0]
    C = np.zeros([dims**2,dims**2])
    for i in range(nSamples):
       c = np.reshape(Gradients[i,:],[1.-1])
       C += (c.T @ c)
    C /= nSamples
   
    T = np.tile(np.eye(dims), dims)
    C = (T @ C) @ T.T
   
    w, v = np.linalg.eig(C)
    
    w = np.abs(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    
    return w,np.abs(v)

def AS_single(Gradients, nModes,dims,mode=1):
    """" Finds active subspace of a single output"""
    nSamples = np.shape(Gradients)[0]
    Grad_mode = Gradients[:,(mode -1)*dims:dims*mode]
    C = np.zeros([dims,dims])
    for i in range(nSamples):
        c = np.reshape(Grad_mode[i,:],[1,-1])
        C += (c.T @ c)
    C = C/nSamples
    
    w, v = np.linalg.eig(C)
    
    w = np.real(w)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    
    return w,np.real(v)

def MaxDisScipy(Eval_points,output_samples=1,constraints = [0.5,1.5]):
    """
    Max distance algorithm using optimiser

    Parameters
    ----------
    Eval_points : numpy array (rows = points, cols = dimension)
        Currently evaluated points
    output_samples : integer, optional
        Number of samples required to be returned. The default is 1.
    constraints : numpy array, optional
        boundaries to next point [ low , high ].  The default is [0.5,1.5].

    Returns
    -------
    S : numpy array (rows = output_samples, cols = dimensions)
        returns points to evaluate that are a maximum distance.

    """
    bounds = (tuple(constraints),)*np.shape(Eval_points)[1]
    
    for i in range(output_samples):
        x0 = constraints[0]+np.random.rand(1,np.shape(Eval_points)[1])*(constraints[1]-constraints[0])
        
        s_new = minimize(MinEuclidDistance,x0,args=(Eval_points),bounds=bounds)
        
        Eval_points = np.append(Eval_points,np.reshape(s_new.x,[1,-1]),axis=0)
        if i == 0:
            S = np.reshape(s_new.x,[1,-1])
            
        else:
            S = np.append(S,np.reshape(s_new.x,[1,-1]),axis=0)
    
    return S

def MinEuclidDistance(y,X):
    """Finds the minimum distance between vector y and set of vectors, X"""
    d = -np.min(np.sum((X-y)**2,axis=1))
    return d   


def ShotgunMaxDis(Eval_points,output_samples=1,constraints = [0.5,1.5],shells=100):
    """
    Uses random samples to explore subspace. Measures distance between these 
    points and evaluated points. Point furthest from all points undergoes 
    second (smaller) resampling. Maximum distanced point becomes next sampled point.
    Repeats until number of sample points added equals output_samples.

    Parameters
    ----------
    Eval_points : Points currently evaluated
    output_samples : number of samples required as output
    constraints : 
    shells : number of LHS samples 

    Returns
    -------
    Batch of sample points (size = output_samples,problem_dimensions)

    """
    
    P,dims = np.shape(Eval_points) # Determines properties of problem
    
    # Create random samples within constraints
    S0 = constraints[0] + (constraints[1]-constraints[0])*lhs(dims,shells) 
    #distancing = []
    for j in range(output_samples):
        maxDis = np.min(np.sum((Eval_points - S0[0,:])**2,axis=1))
        p0 = 0
        
        
        # Intitial round
        for i in range(1,shells):
            Dis = np.min(np.sum((Eval_points - S0[i,:])**2,axis=1))
            if Dis > maxDis:
                maxDis = Dis
                p0 = i
        
        # Second round with samller radii
        S1 = S0[p0,:] + 0.1*(constraints[1]-constraints[0])*lhs(dims,shells)
        S1[S1>constraints[1]] = constraints[1]
        S1[S1<constraints[0]] = constraints[0]
        
        p1 = 0
        
        for k in range(0,shells):
            Dis = np.min(np.sum((Eval_points - S1[k,:])**2,axis=1))
            if Dis > maxDis:
                maxDis = Dis
                p1 = k
        if p1 == 0: S = S0[p0,:]
        else: S = S1[p1,:]
        print(S)
        Eval_points = np.append(Eval_points,S.reshape([1,-1]),axis=0)
        if j == 0: sample = S.reshape([1,-1])
        else: sample = np.append(sample,S.reshape([1,-1]),axis=0)
        #distancing = np.append(distancing,maxDis)
    
    
    return sample



#-----------------------------------------------------------------------------
# Other Functions
def MAC_Error(V1,V2):
    """" Calculates modal assurance criterion (MAC) error between two matrices"""
    nom = np.abs(np.transpose(V1) @ V2)**2
    denom1 = np.diag(np.transpose(V1) @ V1).reshape([-1,1])
    denom2 = np.diag(np.transpose(V2) @ V2).reshape([-1,1])
    denom = denom1 @ np.transpose(denom2)
    MAC = nom/denom 
    return 1-(np.sum(np.diag(MAC)))/np.shape(np.diag(MAC))[0]

def AS_LocTol(Eigenvalues,tolerance):
    ''' Locates active variables based on specified tolerance'''
    
    Eigenvalues /= Eigenvalues[0] #Normalises eigenvalues
    
    return np.argmax(Eigenvalues<tolerance)

def MAC_matrix(V1,V2):
    """" Calculates modal assurance criterion (MAC) error between two matrices"""
    nom = np.abs(np.transpose(V1) @ V2)**2
    denom1 = np.diag(np.transpose(V1) @ V1).reshape([-1,1])
    denom2 = np.diag(np.transpose(V2) @ V2).reshape([-1,1])
    denom = denom1 @ np.transpose(denom2)
    MAC = nom/denom
    return MAC

def MAC_Frob(V1,V2):
    """
    Finds the Frobenius norm between the MAC and identity matrices. 
    
    MAC matrix calculated first: 
        if active subsapces are the same, results in identity matrix
        if different, results in anti-diagonal matrix
    
    MAC can be calculated for mismatching subspace dimensions. This function
    calculates Frobenius norm for the dimension of the diagonal only 
        (the lowest eigenvector number between active subspaces)
    
    Lower output is best.

    Parameters
    ----------
    V1 : 2D Matrix
        First active subspace to be compared
    V2 : 2D Matrix
        Second active subspace to be compared

    Returns
    -------
    FrobErr : Numerical value
        Frobenius norm of (MAC - I), where I is the identity matrix.
    """
    # Calculates MAC matrix
    nom = np.abs(np.transpose(V1) @ V2)**2
    denom1 = np.diag(np.transpose(V1) @ V1).reshape([-1,1])
    denom2 = np.diag(np.transpose(V2) @ V2).reshape([-1,1])
    denom = denom1 @ np.transpose(denom2)
    MAC = nom/denom
    
    # determines shape of identity matrix
    k = np.shape(np.diag(MAC))[0] 
    
    # Difference between MAC and identity matrix
    FrobMat = MAC[:k,:k]-np.eye(k)
    
    return np.linalg.norm(FrobMat,'fro')
    
def Reiman_Distance(V1,V2):
    """
    Calculates the Reimannian distance between two manifolds/active subspaces.

    Parameters
    ----------
    V1 : 2D Matrix
        First active subspace to be compared
    V2 : 2D Matrix
        Second active subspace to be compared

    Returns
    -------
    d : Numerical Value
        DESCRIPTION.

    """
    
    V12 = np.transpose(V1)@V2
    S = np.linalg.svd(V12,compute_uv=False)
    S[S>=1] = int(1)
    d = np.sum(np.arccos(S)**2)**0.5
    
    return d
    

