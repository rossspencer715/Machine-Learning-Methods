
import numpy as np
import mlp
import matplotlib.pyplot as plt

#load data
data = np.load('dataSet.npy')

data[:,[0,1]]

data_in = data[:,[0,1]]
target_in = data[:,[2]]

## used to decide how many hidden layers to use
for j in range(0,5):
    errors = []

    for i in range(1,45):
        print("Hidden layers: ", i, "\n")
        #Set up Neural Network
        
        hidden_layers = i
        NN = mlp.mlp(data_in,target_in,hidden_layers)

        #Analyze Neural Network Performance
        err = NN.mlptrain(data_in,target_in, .1, 10)
        errors.append(err)




    plt.plot(range(1,45), errors, 'ro')
    plt.axis([0, 45, min(errors)-1, max(errors)+1])
plt.show()

learning_rates = [0,.001, .01, .05, .1,.2,.3,.4,.5,.6,.7]#,.8,.9,1,2]
for learning_rate in learning_rates: 
# 4 hidden nodes
    NN = mlp.mlp(data_in,target_in,4)
    NN.mlptrain(data_in,target_in, learning_rate, 10)
#
#outs = NN.mlpfwd(data_in)
#YY=0
#YN=0
#NY=0
#NN=0
#for i in range(0,np.shape(data_in)[0]):
#    if target_in[i] == 1 and outs[i] == 1:
#        YY += 1
#    elif target_in[i] == 1 and outs[i] == 0:
#        YN += 1
#    elif target_in[i] == 0 and outs[i] == 0:
#        NN += 1
#    else:
#        NY += 1
#        JK, confmat already does what I'm trying to...
    print("learning_rate: ",learning_rate,)
    NN.confmat(data_in, target_in)
    
    
iterations = [1,5,10,15,20,50,100,500,1000]
for iteration in iterations: 
# 4 hidden nodes
    NN = mlp.mlp(data_in,target_in,4)
    NN.mlptrain(data_in,target_in, .1, iteration)
    NN.confmat(data_in, target_in)
    
    
###### final decision: 4 hidden, alpha=0.1, 50 iterations    
NN = mlp.mlp(data_in,target_in,4)
NN.mlptrain(data_in,target_in, .1, 50)
NN.confmat(data_in, target_in)
