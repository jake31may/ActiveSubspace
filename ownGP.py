#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:26:04 2023

@author: jakehollins
"""
# Code for own implementation of Gaussian Process

## Original GP

import numpy as np
from numpy import linalg

import scipy as scipy
from scipy.optimize import minimize
from scipy.optimize import brute
from numpy.random import rand

import matplotlib.pyplot as plt
from pyDOE import *

class GP:
    def __init__(self, X_training,y_training,kernel,ARD=False):
        """Initialises model by importing training data (X,y), kernel, with optional automatic relevance determination (ARD)."""
        # Check dimensions of training data
        if np.shape(X_training)[0] != min(np.shape(X_training)):
            X_training = X_training.T
        if np.shape(y_training)[0] != np.shape(X_training)[1]:
            y_training = y_training.T
            
        self.y_bar = np.mean(y_training)
        self.y = y_training - self.y_bar
        self.X = X_training
        self.kernel = kernel
        self.ARD = ARD
        
        if ARD == True: self.hyp_dims = np.shape(self.X)[0]+2
        else: self.hyp_dims = 3
            
    def train(self):
        """Trains model according to initialisation. \n Optimises using scipy L-BFGS algorithm."""
        self.hyperparameters = opt_hyperparameters(self.X,self.y,self.kernel,self.hyp_dims,self.ARD)
        self.K = self.kernel(self.X,self.X,self.hyperparameters,self.ARD)
        print(np.shape(self.K))

    
    def predict(self,x_star):
        """Using the trained model, predict f(x*) values, where x* is the input variables."""
        noise = self.hyperparameters[0]
        
        # Check dimensions of training data
        if np.shape(x_star)[0] != min(np.shape(x_star)):
            x_star = x_star.T
        print('x* shape: ',np.shape(x_star))
        K_star = self.kernel(self.X,x_star,self.hyperparameters,self.ARD)
        Ky = self.K+noise*np.eye(np.shape(self.K)[1])
        invKy = linalg.inv(Ky)
        L = linalg.cholesky(Ky)
        alpha = linalg.solve(np.transpose(L),linalg.solve(L,self.y))
        self.f_star = np.transpose(K_star)@invKy@self.y + self.y_bar
        self.f_starCov = kernel(x_star,x_star,self.hyperparameters,self.ARD) - np.transpose(K_star)@invKy@K_star
        self.LL = -0.5*np.transpose(self.y)@alpha - np.sum(np.log(np.diag(L))) - 0.5* np.shape(self.y)[0]*np.log(2*np.pi)
        self.std = Cov2Std(self.f_starCov)
        
        return self.f_star, self.std

def Squared_Exponential(X1,X2,hyperparameters,ARD=False):
        
        sigma = hyperparameters[1]
        if ARD == False:
            lengthscale = hyperparameters[2]
            exponent = -0.5 * scipy.spatial.distance.cdist(X1.T, X2.T, 'sqeuclidean')/lengthscale**2
        
        else:
            lengthscale = np.linalg.inv(np.diag(hyperparameters[2:]))**2
            print('ell:',lengthscale)
            X1 = lengthscale@X1
            X2 = lengthscale@X2
            exponent = -0.5 * scipy.spatial.distance.cdist(X1.T, X2.T, 'sqeuclidean')
            
        return sigma**2 * np.exp(exponent)

def opt_hyperparameters(X_training,y_training,kernel,hyp_dims,ARD):
    iterations = 100 if ARD == True else 20
    
    LLmax = 1E10
    for i in range(20):
        theta_init = 10000**(lhs(1,hyp_dims)-0.5)
        theta_init[0] = lhs(1,1)
        bounds = np.tile([1E-8,1E12], hyp_dims).reshape(hyp_dims,-1)
        LL = scipy.optimize.minimize(Log_MarginalLikelihood,
                                     theta_init, args=(kernel, X_training, y_training),
                                    bounds = bounds)
        if LL.fun < LLmax:
            theta = LL.x
            LLmax = LL.fun

    return theta

def Log_MarginalLikelihood(hyperparameters,kernel,X,y):
        if hyperparameters[0] < 0: hyperparameters[0] = 1E-3
        noise = hyperparameters[0]
        K = kernel(X,X,hyperparameters)
        Ky = K+noise*np.eye(np.shape(K)[1])
        invKy = linalg.inv(Ky)
        L = linalg.cholesky(Ky)
        alpha = linalg.solve(np.transpose(L),linalg.solve(L,y))
        LL = 0.5*np.transpose(y)@alpha + np.sum(np.log(np.diag(L))) + 0.5* np.shape(y)[0]*np.log(2*np.pi)
        return LL.squeeze()
    
def Cov2Std(CovarianceMatrix):
    return np.sqrt(np.diag(CovarianceMatrix))

def ActiveSubspace(partialGrad):
    
    dims = np.shape(partialGrad)[1]
    nSamples = np.shape(partialGrad)[0]
    S = np.zeros([dims,dims]);
    for i in range(nSamples):
        s = np.array([partialGrad[i,:]])
        S = np.transpose(s)@s + S
    C = S/nSamples
    Lambda,W = np.linalg.eig(C)
    L = np.abs(Lambda)
    index = np.argsort(-L)
    W = W[:,index]
    colour = ['r','g','b']
    
    fig = plt.figure()
    ax = fig.add_subplot(121,yscale='log',xlabel='Eigenvalue Number',ylabel='Eigenvalue')
    ax.bar(range(dims),L[index],color=colour[:dims])
    
    ax2 = fig.add_subplot(122, ylabel='$x_2$',xlabel='$x_1$')
    X,Y = X_train[0,:].T,X_train[1,:].T
    gx,gy = fgrad[:,0],fgrad[:,1]
    for i in range(len(X)):
        ax2.arrow(X[i],Y[i],gx[i]/50,gy[i]/50,width=0.02)
    ax2.scatter(X.squeeze(),Y.squeeze(),color='k')

    Lam = Lambda.T/(np.sum(Lambda))
    ax2.arrow(0.75,0.75,-W[0,0]+0.75,-W[1,0]+0.75,width=0.04,color='r')
    ax2.arrow(0.75,0.75,-W[0,1]+0.75,-W[1,1]+0.75,width=0.04,color='g')
    
    return np.real(W), Lambda[index]

def plotGP3D(pointX,pointY,x_star,f_star,f_std,angle=230):
    f_star = f_star.squeeze()
    pointY = pointY.squeeze()
    pointX = pointX.squeeze()
    f_std = f_std.squeeze()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', xlim=(0.75,1.25), ylim=(0.75,1.25))
    ax.view_init(elev=10., azim=angle)

    mean = [x_star[0,:],x_star[1,:],f_star]
    upperb = [x_star[0,:],x_star[1,:],f_star+2*f_std]
    lowerb = [x_star[0,:],x_star[1,:],f_star-2*f_std ]

    ax.plot(*upperb, lw=2,alpha=0.5,c='green')
    ax.plot(*lowerb, lw=2,alpha=0.5,c='green')
    ax.plot(*mean, lw=2,alpha=0.3,c='blue')
    ax.scatter3D(pointX[0,:],pointX[1,:],pointY,c='red',s = 60, alpha = 1)
    
def plotGP2D(pointX,pointY,X_star,f_star,f_std):
    f_star = f_star.squeeze()
    f_std = f_std.squeeze()
    X_star = X_star.squeeze()

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    
    ax1.fill_between(X_star, f_star-2*f_std, f_star+2*f_std, color='red', 
                     alpha=0.15, label='$Posterior variance \sigma_{2|1}$')
    ax1.plot(X_star, f_star, 'r-', lw=2, label='$Posterior mean, \mu_{2|1}$')
    ax1.plot(pointX.squeeze(), pointY.squeeze(), 'ko', linewidth=2, label='$Initial sample points(x_1, y_1)$')

    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.legend()