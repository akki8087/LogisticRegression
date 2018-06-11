# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:33:12 2018

@author: NP
"""


import pandas as pd
import numpy as np
import math
data = pd.read_csv('ex2data1.txt',names = ['x0','x1','y'])

X = data.iloc[:,:-1]
y = data.iloc[:,-1:]

alpha = 0.01
X.insert(0, 'one', 1, allow_duplicates=True)
n = 1500

theta = np.array([[0.0], [0.0], [0.0]])
lambd = 1000
def GD(X,y,theta,alpha,n,lambd):
    i = 0
    cost = []
    m = len(X)
    while i < n:
        
        y_p = pd.DataFrame(np.dot( X,theta))
        
        y_pred = y_p[0].apply(lambda x: 1/(1 + math.exp(-(int(x))))) 
        
        cost.append(sum(-y['y']*math.log(y_pred[0]) - (1 - y['y'])*math.log(1 - y_pred[0]))/m)

    
        temp0 = theta[0] - (alpha*(sum(y_pred[0] - y['y']))/m)
        temp1 = theta[1]*(1 - alpha*lambd/m) - (alpha*(sum((y_pred[0] - y['y'])*X.iloc[:,1])/m)) 
        
        theta[0] = temp0
        theta[1] = temp1
        
        i += 1
    return theta

GD(X,y,theta,alpha,n,lambd)
