# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author: Ross Spencer
Date:   09/13/18
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N);
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sinc(x) + e
	return x,t

#def plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]):
#    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
#       training data, the true function, and the estimated function'''
#    p1 = plt.plot(x1, t1, 'bo') #plot training data
#    p2 = plt.plot(x2, t2, 'g') #plot true value
#    if(x3 is not None):
#        p3 = plt.plot(x3, t3, 'r') #plot training data
#
#    #add title, legend and axes labels
#    plt.ylabel('t') #label x and y axes
#    plt.xlabel('x')
#    
#    if(x3 is None):
#        plt.legend((p1[0],p2[0]),legend)
#    else:
#        plt.legend((p1[0],p2[0],p3[0]),legend)
        
"""
This seems like a good place to write a function to learn your regression
weights!
    
"""
'''added by ross, finds weights/coefficients for polynomial of order M to fit data:'''
    
def FitData(x,t,M):
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t
    return w
        

""" ======================  Variable Declaration ========================== """

l = 0 #lower bound on x
u = 10 #upper bound on x
N = 20 #number of samples to generate
gVar = .25 #variance of error distribution
M =  6 #regression model order
""" =======================  Generate Training Data ======================= """
data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T

x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

x2 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
t2 = np.sinc(x2) #compute the true function value
    
""" ========================  Train the Model ============================= """

w = FitData(x1, t1, M)
x3 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
X = np.array([x3**m for m in range(w.size)]).T
t3 = X@w #compute the predicted value

#plotData(x1,t1,x2,t2,x3,t3,['Training Data', 'True Function', 'Estimated\nPolynomial'])
#print(w)


""" ======================== Generate Test Data =========================== """


"""This is where you should generate a validation testing data set.  This 
should be generated with different parameters than the training data!   """
   
data_uniform_test  = np.array(generateUniformData(50, l, u, gVar)).T

x4 = data_uniform_test[:,0]
t4 = data_uniform_test[:,1]
print(x4, t4)

x5 = np.arange(0,10,0.001)  #get equally spaced points in the xrange
t5 = np.sinc(x5) #compute the true function value

x6 = np.arange(0,10,0.001)
X_test = np.array([x6**m for m in range(w.size)]).T
t6 = X_test@w #compute the predicted value

#plt.close('all')
#plot for my own testing purposes
#plotData(x4,t4,x5,t5,x6,t6,['Testing Data', 'True Function', 'Estimated\nPolynomial'])


""" ========================  Test the Model ============================== """

""" This is where you should test the validation set with the trained model """

#we want to generate training data, generate val data, loop: fit training data, then find error from val and training


ERMStrain = []
ERMStest = []
for M in range(0,10):
    #fit training data:
    w = FitData(x1,t1,M) 
    
    #get predicted value on training data
    Xtrain = np.array([x1**m for m in range(w.size)]).T
    ttest = Xtrain@w
    
    
    #get predicted value on validation testing data
    X = np.array([x4**m for m in range(w.size)]).T
    t4 = X@w #compute the predicted value
    
    #find E_RMS for training data
    err = 0
    for i in range(len(x1)):        
        err += np.sqrt((ttest[i] - np.sinc(x1[i]))**2)
    ERMStrain.append(err/np.sqrt(len(x1)))

    #find E_RMS for testing data
    err = 0
    for i in range(len(x4)):
        err += np.sqrt((t4[i] - np.sinc(x4[i]))**2)
    ERMStest.append(err/np.sqrt(len(x4)))
    
    
    #now to recreate the plot from the book...
#plt.close('all')
#xerr = range(len(ERMStrain))
#p1 = plt.plot(xerr, ERMStrain, 'b') #plot training data
#p2 = plt.plot(xerr, ERMStest, 'r') #plot true value
#  
##add title, legend and axes labels
#plt.ylabel('$E_{RMS}$') #label x and y axes
#plt.xlabel('M')
#    
#plt.legend((p1[0],p2[0]),['Training Data', 'Testing Data'])


"""
===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """

#True distribution mean and variance 
trueMu = 4
trueVar = 2

#Initial prior distribution mean and variance (You should change these parameters to see how they affect the ML and MAP solutions)
priorMu = 4
priorVar = 15

numDraws = 20 #Number of draws from the true distribution


"""========================== Plot the true distribution =================="""
#plot true Gaussian function
step = 0.01
l = -20
u = 20
x = np.arange(l+step/2,u+step/2,step)
#plt.figure(0)
#p1 = plt.plot(x, norm(trueMu,trueVar).pdf(x), color='b')
#plt.title('Known "True" Distribution')

"""========================= Perform ML and MAP Estimates =================="""
#Calculate posterior and update prior for the given number of draws
''' MLE of our distribution: '''
drawResult = []
ML = []
MAP = []
for draw in range(numDraws):
    #draw from our distribution
    drawResult.append(np.random.normal(trueMu, trueVar, 1)[0])
    #get our MLE value Ybar
    currML = sum(drawResult)/len(drawResult)
    #get our MAP using prior
    currMAP = trueVar/(len(drawResult)*(priorVar) + trueVar)*priorMu + len(drawResult)*(priorVar)/(len(drawResult)*(priorVar) + trueVar)*currML
    
    #save MLE in our array
    ML.append(currML)
    #save MAP in our array
    MAP.append(currMAP)
    
    #find sigma_N
    MLVar = (1/priorVar + len(drawResult)/trueVar)**(-1)
    
    #print our ML and MAP
    print('Frequentist/Maximum Likelihood of Mu:' + str(currML))
    print('Bayesian/MAP Estimate of Mu:' + str(currMAP))
    
    #update the prior distribution with the posterior for next iteration
    priorMu = currML
    priorVar = MLVar
    #input("Hit enter to continue...\n")

"""
You should add some code to visualize how the ML and MAP estimates change
with varying parameters.  Maybe over time?  There are many differnt things you could do!
"""

#this function plots the ML and MAP on the t axis (measured in iterations of the for-loop above)
#def plotMLMAP(ML, MAP):
#    '''plotBeta(a=1,b=1): Plot plot beta distribution with parameters a and b'''
#    xrange = range(0,len(ML))  #get equally spaced points in the xrange
#    plt.figure()
#    plt.plot(xrange, ML, 'b')
#    plt.plot(xrange, MAP, 'r')
#    plt.ylabel('mu-hat')
#    plt.xlabel('t')
#    plt.legend(('Max Likelihood', 'Max A Posteriori'))
#    plt.show()
#    
#plotMLMAP(ML, MAP)
#
