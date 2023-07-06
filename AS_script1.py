#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:00:56 2023

@author: jakehollins
"""
import ActiveSubspace as AS
import numpy as np

model = AS.AS_model()
model.init_sample(samples=5)


#%% Kneedle Algo
model1 = model
W1,V1 = model.FindSubspace()
print('Eigenvalues: \n',W1,'\n EigenVectors: \n',V1)

MAC_err = 1
err_threshold = 0.01
i = 0

while MAC_err > err_threshold and i < 5:
    i += 1
    print('Iteration number: ',i)
    
    model.add_samples(n_samples=5)
    W2,V2 = model.FindSubspace()
    print('Eigenvalues iter 1: \n',W2,'\n EigenVectors iter 1: \n',V2)
    if np.shape(V2) == np.shape(V1):
        MAC_err = AS.MAC_Error(V1,V2)
        print(MAC_err)
    V1 = V2
    

#%% Tolerance
model1 = model
W1,V1 = model1.FindSubspace(CovarianceMatrix=AS_sum,AS_locator='tolerance',tolerance=0.1)
print('Eigenvalues: \n',W1,'\n EigenVectors: \n',V1)
#%%
MAC_err = 1
err_threshold = 0.1
i = 0
while MAC_err > err_threshold and i < 5:
    i += 1
    print('Iteration number: ',i)
    
    model1.add_samples(n_samples=5)
    
    W2,V2 = model1.FindSubspace(CovarianceMatrix=AS_transform,AS_locator='tolerance',tolerance=0.01)
    print('Eigenvalues iter 1: \n',W2,'\n EigenVectors iter 1: \n',V2)
    if np.shape(V2) == np.shape(V1):
        MAC_err = AS.MAC_Error(V1,V2)
        print('MAC Error for iteration ',i,': ',MAC_err)
    else:
        print('AS variables do not match!')
    V1 = V2
  #%% Single Output
  model1 = model
  W1,V1 = model1.FindSubspace(CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.01,single_mode =1)
  print('Eigenvalues: \n',W1,'\n EigenVectors: \n',V1)
  #%%
  MAC_err = 1
  err_threshold = 0.1
  i = 0
  while MAC_err > err_threshold and i < 5:
      i += 1
      print('Iteration number: ',i)
      
      model1.add_samples(n_samples=5)
      
      W2,V2 = model1.FindSubspace()
      print('Eigenvalues iter 1: \n',W2,'\n EigenVectors iter 1: \n',V2)
      if np.shape(V2) == np.shape(V1):
          MAC_err = AS.MAC_Error(V1,V2)
          print('MAC Error for iteration ',i,': ',MAC_err)
      else:
          print('AS variables do not match!')
      V1 = V2
    
    #%%
    MAC_err = 1
    err_threshold = 0.1
    i = 0
    while MAC_err > err_threshold and i < 5:
        i += 1
        print('Iteration number: ',i)
        
        model1.add_samples(n_samples=5)
        
        W2,V2 = model1.FindSubspace(CovarianceMatrix=AS_transform,AS_locator='tolerance',tolerance=0.001)
        print('Eigenvalues iter 1: \n',W2,'\n EigenVectors iter 1: \n',V2)
        if np.shape(V2) == np.shape(V1):
            MAC_err = AS.MAC_Error(V1,V2)
            print('MAC Error for iteration ',i,': ',MAC_err)
        else:
            print('AS variables do not match!')
        V1 = V2
        
        
    