#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:00:56 2023

@author: jakehollins
"""
import ActiveSubspace as AS

model = AS.AS_model()
#%%
model.init_sample(samples=5)

#%%

W1,V1 = model.FindSubspace()
print('Eigenvalues: \n',W1,'\n EigenVectors: \n',V1)

#%%

MAC_err = 1
err_threshold = 0.1
i = 0

while MAC_err > err_threshold and i < 5:
    i += 1
    print(i)
    
    model.add_samples(n_samples=5)
    W2,V2 = model.FindSubspace()
    print('Eigenvalues iter 1: \n',W2,'\n EigenVectors iter 1: \n',V2)
    MAC_err = AS.MAC_Error(V1,V2)
    print(MAC_err)
    


