#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:19:51 2023

@author: jakehollins
"""
# Practising GP



from numpy.random import rand
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt

seed(12)

x_train = rand(1,20)*4
print(x_train)
x_true = np.linspace(0,4)

def eq1(x):
    return (x)**3 + 2*np.sin(x)**2  - 13*x + 10

y_train = eq1(x_train)

y_train = y_train + (rand(1,20)-0.5)*3
y_true = eq1(x_true)

print(y_train)

kernel = Squared_Exponential
model = GP(x_train,y_train,kernel)

model.train()

x_test = np.array([np.linspace(0,4)])

f_star, f_std = model.predict(x_test)

plotGP2D(x_train,y_train,x_test,f_star,f_std)
plt.plot(x_true,y_true)

