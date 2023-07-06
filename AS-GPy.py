#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:02:26 2023

@author: jakehollins
"""
import GPy
import ActiveSubspace as AS

import matplotlib.pyplot as plt


def RMSE(x,x_star):
    return np.sum((x-x_star)**2)/np.shape(x)[0]

#%%

HaltonAS = AS.AS_model(dims=5)
HaltonAS.init_sample(n_samples=5,sample_range=[0.5,1.5],sampler='Halton')

for loop in range(9):
    print(loop)
    HaltonAS.add_samples(n_samples=5)

#%%
mode = 6
test_f = np.loadtxt("2023_Apr_03-Freqs.csv",delimiter=',')
test_y = test_f[:,mode-1].reshape([-1,1])
test_X = np.loadtxt("2023_Apr_03-ps.csv",delimiter=',')

MSE_Halton = []
for i in range(49):
    X = HaltonAS.samples[:(int(i)+1),:]
    y = HaltonAS.Frequencies[:(int(i)+1),mode-1].reshape([-1,1])
    
    kernel = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X,y,kernel)
    
    m.optimize_restarts(num_restarts = 10)
    y_star = m.predict(test_X)
    
    y_star = np.array(y_star).squeeze()
    mean_y = y_star[0,:].reshape([-1,1])
    MSE_Halton = np.append(MSE_Halton,MSE(y,mean_y))

#%%

MaxDisAS = AS.AS_model(dims=5)
MaxDisAS.init_sample(n_samples=5,sample_range=[0.5,1.5],sampler='lhs')

for loop in range(9):
    print(loop)
    MaxDisAS.add_samples(n_samples=5,sampler='MaxDisOpt')

#%%
MSE_MaxDis = []
for i in range(49):
    X = MaxDisAS.samples[:(int(i)+1),:]
    y = MaxDisAS.Frequencies[:(int(i)+1),mode-1].reshape([-1,1])
    
    kernel = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X,y,kernel)
    
    m.optimize_restarts(num_restarts = 10)
    y_star = m.predict(test_X)
    
    y_star = np.array(y_star).squeeze()
    mean_y = y_star[0,:].reshape([-1,1])
    MSE_MaxDis = np.append(MSE_MaxDis,MSE(y,mean_y))

#%%
MSE_ROM= []
for i in range(49):
    X = ROmodel.samples[:(int(i)+1),:]
    y = ROmodel.Frequencies[:(int(i)+1),mode-1].reshape([-1,1])
    
    kernel = GPy.kern.RBF(input_dim=5, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X,y,kernel)
    
    m.optimize_restarts(num_restarts = 10)
    y_star = m.predict(test_X)
    
    y_star = np.array(y_star).squeeze()
    mean_y = y_star[0,:].reshape([-1,1])
    MSE_ROM = np.append(MSE_ROM,MSE(y,mean_y))


#%%
mode = 6
Gradient = np.loadtxt("2023_Apr_03-grad.csv",delimiter=',')
test_y = test_f[:,mode-1].reshape([-1,1])
MSE_Halton = []
for i in range(49):
    X = HaltonAS.samples[:(int(i)+1),:]
    y = HaltonAS.Frequencies[:(int(i)+1),mode-1].reshape([-1,1])
    
    kernel = GPy.kern.Matern32(input_dim=5, variance=1., lengthscale=1.,ARD=True)
    m = GPy.models.GPRegression(X,y,kernel)
    
    m.optimize_restarts(num_restarts = 10)
    y_star = m.predict(test_X)
    
    y_star = np.array(y_star).squeeze()
    mean_y = y_star[0,:].reshape([-1,1])
    MSE_Halton = np.append(MSE_Halton,MSE(test_y,mean_y))

MSE_MaxDis = []
for i in range(49):
    X = MaxDisAS.samples[:(int(i)+1),:]
    y = MaxDisAS.Frequencies[:(int(i)+1),mode-1].reshape([-1,1])
    
    kernel = GPy.kern.Matern32(input_dim=5, variance=1., lengthscale=1.,ARD=True)
    m = GPy.models.GPRegression(X,y,kernel)
    
    m.optimize_restarts(num_restarts = 10)
    y_star = m.predict(test_X)
    
    y_star = np.array(y_star).squeeze()
    mean_y = y_star[0,:].reshape([-1,1])
    MSE_MaxDis = np.append(MSE_MaxDis,MSE(test_y,mean_y))

#%%

eign = np.linspace(1,5,5)
init_samps = 5
add_samps = 1
colors = ['r','b','g']
i = 0
its =[51,51]
#m2c = Shotgun
plt.figure(figsize=(18,10))
title = 'Active Subspace errors for mode ' + str(mode)
plt.suptitle(title)
plt.subplot(2,1,1)


for m2c in [HaltonAS,MaxDisAS]:
    
    # True Active subsapce from 200 samples
    [D,V] = AS.AS_single(Gradients=Gradient,nModes=nModes,dims=5,mode=mode)
    k = KneeLocator(eign,D,direction='decreasing',curve='convex').knee
    t1 = AS.AS_LocTol(D,0.01)
    t2 = AS.AS_LocTol(D,0.001)
    Vknee = V[:,:int(k)]
    Vtol1 = V[:,:t1]
    Vtol2 = V[:,:t2]
    
    # Active subspace for initial samples
    [Wk,Vk] = m2c.FindSubspace(Gradients = m2c.Gradients[:init_samps,:],CovarianceMatrix=AS_single,AS_locator='knee',single_mode=mode)
    [Wt1,Vt1] = m2c.FindSubspace(Gradients = m2c.Gradients[:init_samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.01,single_mode=mode)
    [Wt2,Vt2] = m2c.FindSubspace(Gradients = m2c.Gradients[:init_samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.001,single_mode=mode)
    
    s_k = np.shape(Vk)[1]
    s_t1 = np.shape(Vt1)[1]
    s_t2 = np.shape(Vt2)[1]
    
    Ek = AS.Reiman_Distance(Vknee,Vk)
    Et1 = AS.Reiman_Distance(Vtol1,Vt1)
    Et2 = AS.Reiman_Distance(Vtol2,Vt2)
    
    EkR = []
    Et1R = []
    Et2R = []
    
    vals = np.floor(np.shape(m2c.samples)[0]/add_samps)
    for loop in range(int(vals)):
        #model1.add_samples(n_samples=5)
        samps = (loop+2)*add_samps
        # Active subspace for additional samples
        [Wka,Vka] = m2c.FindSubspace(m2c.Gradients[:samps,:],CovarianceMatrix=AS_single,AS_locator='knee',single_mode=mode)
        [Wt1a,Vt1a] = m2c.FindSubspace(m2c.Gradients[:samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.01,single_mode=mode)
        [Wt2a,Vt2a] = m2c.FindSubspace(m2c.Gradients[:samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.001,single_mode=mode)
            
        EkR = np.append(EkR,AS.Reiman_Distance(Vka,Vk))
        Et1R = np.append(Et1R,AS.Reiman_Distance(Vt1a,Vt1))
        Et2R = np.append(Et2R,AS.Reiman_Distance(Vt2a,Vt2))    
        
        Vt1, Vt2, Vk = Vt1a, Vt2a, Vka
        
        s_k = np.append(s_k,np.shape(Vk)[1])
        s_t1 = np.append(s_t1,np.shape(Vt1)[1])
        s_t2 = np.append(s_t2,np.shape(Vt2)[1])
        
        Ek = np.append(Ek,AS.Reiman_Distance(Vknee,Vk))
        Et1 = np.append(Et1,AS.Reiman_Distance(Vtol1,Vt1))
        Et2 = np.append(Et2,AS.Reiman_Distance(Vtol2,Vt2))
            
    #plt.semilogy(range(int(its[i])),Ek,':',color=colors[i])     # knee
    plt.semilogy(range(int(its[i])),Et1,'-',color=colors[i])     # tol = 0.01
    plt.semilogy(range(int(its[i])),Et2,'--',color=colors[i])    # tol = 0.001
    i+=1


plt.rc('font', size=16)
title = 'Comparison of sampling algorithms with different active subspace threshold mechanic. Mode = ' + str(mode)
plt.xlabel('Samples')
plt.ylabel('Reimannian Distance')
plt.title(title)
#plt.legend(['Knee - Random','tol = 0.01 - Random','tol = 0.001 - Random','Knee - Max. Distance','tol = 0.01 - Max. Distance','tol = 0.001 - Max. Distance',
#            'Knee - Halton','tol = 0.01 - Halton','tol = 0.001 - Halton'])
plt.legend(['tol = 0.01 - Halton','tol = 0.001 - Halton','tol = 0.01 - Max. Distance','tol = 0.001 - Max. Distance',
            'tol = 0.01 - Halton','tol = 0.001 - Halton'])


plt.subplot(2,1,2)
plt.semilogy(range(49),MSE_Halton)
plt.semilogy(range(49),MSE_MaxDis)
title = 'Mean square error for train GP for mode'+str(mode)
plt.xlabel('~Samples used')
plt.ylabel('Mean Square Error')
plt.legend(['Halton GP','Maximum Distance GP'])
file = 'GP_M'+str(mode)+'.png'
plt.savefig(file, dpi=300)

#%%
import numpy as np

mode = 6
test_y = test_f[:,mode-1].reshape([-1,1])

eign = np.linspace(1,5,5)
init_samps = 5
add_samps = 5

MSE_x1 = []
MSE_x2 = []

for m2c in [MaxDisAS]:
    
    # Active subspace for initial samples
    #[Wk,Vk] = m2c.FindSubspace(Gradients = m2c.Gradients[:init_samps,:],CovarianceMatrix=AS_single,AS_locator='knee',single_mode=mode)
    [Wt1,Vt1] = m2c.FindSubspace(Gradients = m2c.Gradients[:init_samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.01,single_mode=mode)
    [Wt2,Vt2] = m2c.FindSubspace(Gradients = m2c.Gradients[:init_samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.001,single_mode=mode)
    
    
    x1 = m2c.samples[:init_samps,:] @ Vt1.reshape([5,-1])
    x2 = m2c.samples[:init_samps,:] @ Vt2.reshape([5,-1])
    y = m2c.Frequencies[:init_samps,mode-1].reshape([-1,1])
    
    kernel1 = GPy.kern.Matern52(input_dim=np.shape(Vt1.reshape([5,-1]))[1], variance=1., lengthscale=1.,ARD=True)
    GP_x1 = GPy.models.GPRegression(x1,y,kernel1)
    GP_x1.optimize_restarts(num_restarts = 10)
    
    kernel2 = GPy.kern.Matern52(input_dim=np.shape(Vt2.reshape([5,-1]))[1], variance=1., lengthscale=1.,ARD=True)
    GP_x2 = GPy.models.GPRegression(x2,y,kernel2)
    GP_x2.optimize_restarts(num_restarts = 10)
    
    test_X1 = test_X @ Vt1.reshape([5,-1])
    test_X2 = test_X @ Vt2.reshape([5,-1])
    
    y_star1 = GP_x1.predict(test_X1)
    y_star2 = GP_x2.predict(test_X2)
    
    y_star1 = np.array(y_star1).squeeze()
    y_star2 = np.array(y_star2).squeeze()
    
    mean_y1 = y_star1[0,:].reshape([-1,1])
    mean_y2 = y_star2[0,:].reshape([-1,1])
    MSE_x1 = np.append(MSE_x1,MSE(test_y,mean_y1))
    MSE_x2 = np.append(MSE_x2,MSE(test_y,mean_y2))
    
    vals = np.floor(np.shape(m2c.samples)[0]/add_samps)
    for loop in range(int(vals)):
        #model1.add_samples(n_samples=5)
        samps = (loop+2)*add_samps
        # Active subspace for additional samples
        #[Wka,Vka] = m2c.FindSubspace(m2c.Gradients[:samps,:],CovarianceMatrix=AS_single,AS_locator='knee',single_mode=mode)
        [Wt1,Vt1] = m2c.FindSubspace(m2c.Gradients[:samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.01,single_mode=mode)
        [Wt2,Vt2] = m2c.FindSubspace(m2c.Gradients[:samps,:],CovarianceMatrix=AS_single,AS_locator='tolerance',tolerance=0.001,single_mode=mode)
            
        x1 = m2c.samples[:samps,:] @ Vt1.reshape([5,-1])
        x2 = m2c.samples[:samps,:] @ Vt2.reshape([5,-1])
        y = m2c.Frequencies[:samps,mode-1].reshape([-1,1])
        
        kernel1 = GPy.kern.Matern52(input_dim=np.shape(Vt1.reshape([5,-1]))[1], variance=1., lengthscale=1.,ARD=True)
        GP_x1 = GPy.models.GPRegression(x1,y,kernel1)
        GP_x1.optimize_restarts(num_restarts = 10)
        
        kernel2 = GPy.kern.Matern52(input_dim=np.shape(Vt2.reshape([5,-1]))[1],ARD=True)
        GP_x2 = GPy.models.GPRegression(x2,y,kernel2)
        GP_x2.optimize_restarts(num_restarts = 10)
        
        
        test_X1 = test_X @ Vt1.reshape([5,-1])
        test_X2 = test_X @ Vt2.reshape([5,-1])
        
        y_star1 = GP_x1.predict(test_X1)
        y_star2 = GP_x2.predict(test_X2)
        
        y_star1 = np.array(y_star1).squeeze()
        y_star2 = np.array(y_star2).squeeze()
        
        mean_y1 = y_star1[0,:].reshape([-1,1])
        mean_y2 = y_star2[0,:].reshape([-1,1])
        MSE_x1 = np.append(MSE_x1,MSE(test_y,mean_y1))
        MSE_x2 = np.append(MSE_x2,MSE(test_y,mean_y2))

#%%
plt.semilogy(range(11),MSE_H_x1)
plt.semilogy(range(11),MSE_H_x2)
plt.semilogy(range(11),MSE_x1)
plt.semilogy(range(11),MSE_x2)
plt.legend(['Halton, tol = 0.01','Halton, tol=0.001','Max Dis, tol = 0.01','Max Dis, tol=0.001'])
plt.xlabel('iterations of 5')
plt.ylabel('MSE')
plt.title('Mode 6')
file = 'GP_Matern32_mode_'+str(mode)+'.png'
plt.savefig(file, dpi=300)

#%%
MSE_H_x1 = MSE_x1
MSE_H_x2 = MSE_x2

#%%

from IPython.display import display
display(GP_x1)

