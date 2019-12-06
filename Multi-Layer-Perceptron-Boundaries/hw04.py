# -*- coding: utf-8 -*-
"""
File:   hw04.py
Author: 
Date:   
Desc:   
    
"""


'''==========QUESTION 1:=========='''
''' first cell in layer 1 '''
def A(x,y):
    z = (-1)*x+2*(y)+5
    return z

''' second cell in layer 1 '''
def B(x,y):
    z = (-.5)*x+(-.1)*(y)+2
    return z

A(1,-1)
B(1,-1)

'''dataset'''
X = [[1,-1],[-2,-3],[-5,-1],[-2,2],[6,-2],[4,3]]

'''activation function'''
def reLu(x):
    if x>0:
        return x
    else:
        return 0

'''find the sum going into A and B'''
for i in range(len(X)):
    print("\t",i)
    print(A(X[i][0], X[i][1]))
    print(B(X[i][0], X[i][1]))
    print("~~~~~~~~")
    
'''cell in layer 2'''
def C(x,y):
    z = .5*x+(-.7)*(y)
    return z
'''print sum going into C with the values passed into it from the activation function'''
for i in range(len(X)):
    print("\t",i)
    print(
            C(reLu(A(X[i][0], X[i][1])), reLu(B(X[i][0], X[i][1])))
            )
    print("~~~~~~~~")

'''print the output of C, aka our y coordinates'''
print("=================================")
for i in range(len(X)):
    print("\t","y",i+1)
    print(
            reLu( C( reLu(A(X[i][0], X[i][1])), reLu(B(X[i][0], X[i][1])) ) )
            )
    print("~~~~~~~~")
    

'''==========QUESTION 2:=========='''
import numpy as np
import matplotlib.pyplot as plt
dataset1 = np.load('dataSet1.npy')
dataset2 = np.load('dataSet2.npy')
dataset3 = np.load('dataSet3.npy')
print(dataset1.shape) #400x3, x1 x2 class
print(dataset2.shape) #300x3, x1 x2 class
print(dataset3.shape) #120x3, x1 x2 class


''' Dataset 1 '''
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset1[:,0], dataset1[:,1], c=dataset1[:,2])
plt.show


'''activation step function'''
def step(x):
    if x>0:
        return 1
    else:
        return 0
def Cell1(x): #X is 1x2
    return step(1*x[0]+0*x[1]-.5)  ##returns 0 if 1*x[0]+0*x[1]-.5<=0 aka x1<=.5, returns 1 else

def Cell2(x): #X is 1x2
    return step(1*x[0]+0*x[1]-1.5)  ##returns 0 if 0*x[0]+1*x[1]-1.5<=0 aka x1<=1.5, returns 1 else

def NNet1(X):
    arr = []
    for i in range(len(X)):
        arr.append(step(Cell1(X[i])-Cell2(X[i])))
        ## CASES: cell1 is 0 (aka x1<=.5), cell2 is 0 (aka x1<=1.5), classify as 0
        ##        cell1 is 1 (aka x1>.5), cell2 is 1 (aka x1>1.5), classify as 0
        ##        cell1 is 1 (aka x1>.5), cell2 is 0 (aka x1<=1.5), classify as 1
        #### being past .5 on the x1 axis is a positive trait, but being past 1.5 on the x1 axis is just as negative a trait and should cancel it out
    return arr
     
labelsTest1 = NNet1(dataset1)
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset1[:,0], dataset1[:,1], c=labelsTest1)
plt.show



def Cell3(x): #X is 1x2
    return step(0*x[0]+1*x[1]-.5)  ##returns 0 if 1*x[0]+0*x[1]-.5<=0 aka x1<=.5, returns 1 else

def Cell4(x): #X is 1x2
    return step(0*x[0]+1*x[1]-1.5)  ##returns 0 if 0*x[0]+1*x[1]-1.5<=0 aka x1<=1.5, returns 1 else

def NNet2(X):
    arr = []
    for i in range(len(X)):
        arr.append(step(Cell3(X[i])-Cell4(X[i])))
        ## CASES: cell1 is 0 (aka x2<=.5), cell2 is 0 (aka x2<=1.5), classify as 0
        ##        cell1 is 1 (aka x2>.5), cell2 is 1 (aka x2>1.5), classify as 0
        ##        cell1 is 1 (aka x2>.5), cell2 is 0 (aka x2<=1.5), classify as 1
        #### being above .5 on the x2 axis is a positive trait, but being above 1.5 on the x2 axis cancels it out
    return arr
     
labelsTest2 = NNet2(dataset1)
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset1[:,0], dataset1[:,1], c=labelsTest2)
plt.show




''' Dataset 2: '''
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset2[:,0], dataset2[:,1], c=dataset2[:,2])
plt.show


def Cell5(x): #X is 1x2
    return step(1*x[0]+0*x[1])  ##returns 0 if 1*x[0]+0*x[1]<=0 aka x1<=0, returns 1 else

def Cell6(x): #X is 1x2
    return step(.05*x[0]+1*x[1]-.5)  ##returns 0 if 0*x[0]+1*x[1]-1.5<=0 aka x1<=1.5, returns 1 else

#not sure if this is a kosher activation function but makes sense to me
## heavily weights if the x2 val is above .5, as that definitely makes it cyan
def split(x):
    if x>=500:
        return 'c'
    elif x>0:
        return 'm'
    else:
        return 'y'

def NNet3(X):
    arr = []
    for i in range(len(X)):
        arr.append(split(40*Cell5(X[i])+500*Cell6(X[i])))
        #### We want the dividing line of x1 being above .5 to be the most heavily weighted characteristic of the cyan class
        #### after that, draw a line between the other 2 on the x1 axis dividing the other 2 classes
    return arr
     
labelsTest3 = NNet3(dataset2)
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset2[:,0], dataset2[:,1], c=labelsTest3)
plt.show




def Cell7(x): #X is 1x2
    return step(1*x[0]+0*x[1]-.5)  ##returns 0 if 1*x[0]+0*x[1]-.5<=0 aka x1<=.5, returns 1 else

def Cell8(x): #X is 1x2
    return step(0*x[0]+1*x[1]-.5)  ##returns 0 if 0*x[0]+1*x[1]-1.5<=0 aka x1<=1.5, returns 1 else

#not sure if this is a kosher activation function but makes sense to me
## heavily weights if the x2 val is above .5, as that definitely makes it cyan
def split2(x):
    if x>=500:
        return 'm'
    elif x>0:
        return 'c'
    else:
        return 'y'

def NNet4(X):
    arr = []
    for i in range(len(X)):
        arr.append(split2(500*Cell7(X[i])+40*Cell8(X[i])))
        #### We want being above .5 to be the most positive feature for the cyan class
        #### For the others, divide it by x2 being above/below .5
    return arr
     
labelsTest4 = NNet4(dataset2)
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset2[:,0], dataset2[:,1], c=labelsTest4)
plt.show



''' Dataset 3: '''
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset3[:,0], dataset3[:,1], c=dataset3[:,2])
plt.show

def Cell9(x): #X is 1x2
    return step(-1*x[0]+0*x[1]+1.5)  ##returns 0 if -1*x[0]+0*x[1]+1.5<=0 aka x1>=1.5, returns 1 else

def Cell10(x): #X is 1x2
    return step(1*x[0]+0*x[1]-2.75)  ##returns 0 if 1*x[0]+0*x[1]-2.75<=0 aka x1<=2.75, returns 1 else

def Cell11(x): #X is 1x2
    return step(1*x[0]+0*x[1]-3.5)  ##returns 0 if 1*x[0]+0*x[1]-3.5<=0 aka x1<=3.5, returns 1 else

'''also want to cut out from x1=2.7 and x1+x2<6'''


def NNet5(X):
    arr = []
    for i in range(len(X)):
        arr.append(step(4*Cell9(X[i])+1*Cell10(X[i])-1*Cell11(X[i])))
        #### make x1 being <1.5 the heaviest weighted characteristic
        #### then x1 being >3.5 should cancel out it being >2.75
    return arr

labelsTest5 = NNet5(dataset3)
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset3[:,0], dataset3[:,1], c=labelsTest5)
plt.show



def Cell12(x): #X is 1x2
    return step(-1*x[0]+0*x[1]+1.5)  ##returns 0 if -1*x[0]+0*x[1]+1.5<=0 aka x1>=1.5, returns 1 else

def Cell13(x): #X is 1x2
    return step(0*x[0]+1*x[1]-1)  ##returns 0 if 1*x[1]-1<=0 aka x2<=1, returns 1 else //want to cut out cloud below 1

def Cell14(x): #X is 1x2
    return step(-1*x[0]+0*x[1]+2.5)  ##returns 0 if 1*x[0]+0*x[1]-2.5<=0 aka x1>=2.5, returns 1 else

def Cell15(x): #X is 1x2
    return step(1*x[0]+0*x[1]-3.5)  ##returns 0 if 1*x[0]+0*x[1]-3.5<=0 aka x1<=3.5, returns 1 else ## want to take even more away than Cell15 if it's past 3.5^^ 


def NNet6(X):
    arr = []
    for i in range(len(X)):
        arr.append(step(50*Cell12(X[i])+5*Cell13(X[i])-6*Cell14(X[i])-20*Cell15(X[i])))
        ## draws most important line at x1=1.5, then x1>3.5 or x1>2.5 should take away from x2 being >1
    return arr

labelsTest6 = NNet6(dataset3)
fig = plt.figure(figsize=(8,8))
plt.scatter(dataset3[:,0], dataset3[:,1], c=labelsTest6)
plt.show