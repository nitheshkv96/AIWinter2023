# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:13:05 2023

@author: NitheshV
"""

#%% Importing essential libraries
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
rng = np.random.default_rng()
warnings.filterwarnings('ignore')
np.random.seed(seed=1)
file = ['optdigits_train.dat', 'optdigits_test.dat', 'optdigits_trial.dat']

#%%
class NNet:
    def __init__(self, inSize, outSize, file = file, k = 3):
        self.inSize = inSize
        self.outSize = outSize
        self.weights, self.out, self.delta = {}, {}, {}
        self.x_train, self.y_train = self.dataExtract(file[0])
        self.x_test, self.y_test = self.dataExtract(file[1])
        self.x_trial, self.y_trial = self.dataExtract(file[2])
        self.k = k
        self.ksize = int(len(self.x_train)/self.k)
        self.yhat = []
        self.outLayer = None
        self.xtrain, self.ytrain = [],[]
    
    
    def initialize(self, hidL, hidU):
        self.weights = {}
        hidL = range(1,hidL+1)
        currXsize = self.inSize
        if len(hidL) != 0 and hidU:
            for l,n in zip(hidL, hidU):
                w = np.random.uniform(-1,1,(currXsize,n))
                b = np.random.uniform(-1,1, n)
                self.weights[l] = np.vstack((w,b))
                currXsize = n
        w = np.random.uniform(-1,1,(currXsize, self.outSize))
        b = np.random.uniform(-1,1, self.outSize)
        self.weights[len(hidL) + 1] = np.vstack((w,b))
        self.outLayer = len(self.weights)
        
        
    def dataExtract(self, file):
        with open(file, 'r') as f:
            data = f.readlines()
        f.close()
        data = np.vstack([np.array(x.split(), dtype = 'int') for x in data])
        return data[:,:-1], data[:,-1]
    
    
    def hotEncoding(self, y):
        outOH = []
        for yo in y:
            out = np.zeros(self.outSize)
            out[yo] = 1
            outOH.append(out)
        return outOH
    
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    

    def forward(self,x):
        currX = x
        b = np.array([1]*len(x)).reshape(len(x),1)
        for k,w in self.weights.items():
            currY = np.dot(currX, w[:-1,:]) + w[-1,:]
            if k < self.outLayer:
                currX = self.sigmoid(currY)
                self.out[k] = np.concatenate((currX, b), axis=1)
            else:
                currX = self.sigmoid(currY)
                self.out[k] = currX  
        self.out[0] = np.concatenate((x, b), axis=1)
    

    def backprop(self, y):
        layer = self.outLayer
        y = self.hotEncoding(y)
        self.delta[layer] = np.multiply(np.multiply(self.out[layer],(1 - self.out[layer])),(self.out[layer] - np.array(y)))
        while layer > 1:
            currW = self.weights[layer]
            temp1 = np.multiply(self.out[layer-1],(1 - self.out[layer-1]))
            if layer == len(self.weights):
                temp2 = np.matmul(self.delta[layer], currW.T)
            else:
                temp2 = np.matmul(self.delta[layer][:,:-1], currW.T)
            self.delta[layer-1] = np.multiply(temp1 , temp2)
            layer -=1
      
    
    def gradient(self,lr,y): 
        self.backprop(y)
        m = len(y)
        for k,w in self.weights.items():
            if k < self.outLayer:
                dw = lr*np.matmul(self.out[k-1].T, self.delta[k][:,:-1])/m
                self.weights[k] -= dw
            else:
                dw = lr*np.matmul(self.out[k-1].T, self.delta[k])/m
                self.weights[k] -= dw
                
    
    def crossValidationSetGenerator(self,i,x,y):
        if i < self.k-1:
            self.xtrain = np.delete(x, np.s_[self.ksize*i:self.ksize*(i+1)],0)
            self.ytrain = np.delete(y, np.s_[self.ksize*i:self.ksize*(i+1)],0)
            self.xval = x[self.ksize*i:self.ksize*(i+1),:]
            self.yval = y[self.ksize*i:self.ksize*(i+1)]
        else:
            self.xtrain = np.delete(x, np.s_[self.ksize*i:],0)
            self.ytrain = np.delete(y, np.s_[self.ksize*i:],0)
            self.xval = x[self.ksize*i:, :]
            self.yval = y[self.ksize*i:]
         
        
    def misClassError(self,y, yhat):
        err = np.sum(y != yhat)/len(y)
        return err
    
    
    def proxyError(self,y,yhat):
        y = self.hotEncoding(y)
        return 0.5*np.mean(np.mean(np.power((y - yhat),2), axis = 1))
    
    
    def modelEvaluate(self,x,y):
        self.forward(x)
        yhat = self.out[self.outLayer]
        y_pred = np.argmax(yhat, axis = 1)
        return [self.misClassError(y, y_pred), self.proxyError(y,yhat)]
    
    
    def predict(self, bestHypSet, modellayout, hyperSet, epochs):
        l, u , lr = bestHypSet
        self.ValErr, self.TrainErr = [],[]
        if l == 1 or l == 0:
            self.initialize(int(l), [int(u)]*int(l))
        else:
            self.initialize(int(l), [int(u[0]),int(u[1])])
        for epoch in range(epochs):
            self.forward(self.x_train)
            self.gradient(lr, self.y_train)
            self.TrainErr.append(self.modelEvaluate(self.x_train, self.y_train))
            self.ValErr.append(self.modelEvaluate(self.x_test, self.y_test))
            
        print('--> Training Errors(MisClassification and Proxy) with Best Model Table:')
        print('------------------------------------------')
        print ("{:<8} : {:<6} : {:<10}".format('Sample #','MCErr', 'ProxyErr'))
        print('------------------------------------------')
        for i in range(len(self.TrainErr)):
            print ("{:<8} : {:.4f} : {:<17}".format(i, self.TrainErr[i][0],self.TrainErr[i][1]))
        print('------------------------------------------')
        print('\n')
        print('--> Testing Errors(MisClassification and Proxy) with Best Model Table:')
        print('------------------------------------------')
        print ("{:<8} : {:<6} : {:<10}".format('Sample #','MCErr', 'ProxyErr'))
        print('------------------------------------------')
        for i in range(len(self.ValErr)):
            print ("{:<8} : {:.4f} : {:<17}".format(i, self.ValErr[i][0],self.ValErr[i][1]))
        print('------------------------------------------')
        
        self.plotError(hyperSet, epochs,modellayout, retrain = True)
        TrainMisClassErr,TrainProxyErr = self.modelEvaluate(self.x_train, self.y_train)
        print('\n ==> Trained Model Performance:')
        print('Training Errors after Model selection:')
        print('Misclassification Error:',TrainMisClassErr)
        print('Proxy Error:', TrainProxyErr)
        print('\n')
        TestMisClassErr,TestProxyErr = self.modelEvaluate(self.x_test, self.y_test)
        print('Test Errors after Model selection:')
        print('Misclassification Error:',TestMisClassErr)
        print('Proxy Error:', TestProxyErr)
        print('\n')
        

    def plotError(self, hyperSet, epochs, modellayout, retrain = False):
        if retrain ==False:
            xaxis = hyperSet
            fig, ax = plt.subplots()
            fig1, ax1 = plt.subplots()
            fig.tight_layout()
            fig1.tight_layout()
            fig1.dpi, fig.dpi = 500, 500
            if type(hyperSet[0]) != str:
                ax.set_xlabel('Learning Rate')
                ax1.set_xlabel('Learning Rate')
                ax.set_xscale('log',base=4)
                ax1.set_xscale('log',base=4)
            else:
                ax.set_xlabel('HyperParameters:')
                ax1.set_xlabel('HyperParameters:')
                fig.subplots_adjust(bottom=0.5)
                fig1.subplots_adjust(bottom=0.5)
                ax.set_xticklabels(hyperSet, fontsize = 4, rotation = 90)
                ax1.set_xticklabels(hyperSet, fontsize = 4, rotation = 90)
            
            ax.set_title('Training and Validation Misclassification losses (Model Selection/CV) \n {}'.format(modellayout))
            ax1.set_title('Training and Validation Proxy Error (Model Selection/CV) \n {}'.format(modellayout))
        else:
            xaxis = np.arange(1,epochs+1)
            fig, ax = plt.subplots()
            ax.set_xlabel('Epochs')
            fig1, ax1 = plt.subplots()
            ax1.set_xlabel('Epochs')
            ax.set_title('Training and Validation Misclassification losses(Retrained Best Model) \n {}'.format(modellayout))
            ax1.set_title('Training and Validation Proxy Error(Retrained Best Model) \n {}'.format(modellayout))
        
        ax.plot(xaxis, np.array(self.ValErr)[:,0],'r--', label = "Validation")
        ax.plot(xaxis, np.array(self.TrainErr)[:,0],'y', label = "Training")
        ax.legend()
        ax.set_ylabel('Losses')
        
        ax1.plot(xaxis, np.array(self.ValErr)[:,1],'r--', label = "Validation")
        ax1.plot(xaxis, np.array(self.TrainErr)[:,1],'y', label = "Training")
        ax1.legend()
        ax1.set_ylabel('Losses')
        
     
    def evaluateTrial(self):
        self.forward(self.x_trial)
        yhat = self.out[self.outLayer]
        y_pred = np.argmax(yhat, axis = 1)
        print('==> Trial Data Prediction Table:')
        print('------------------------------------------')
        print ("{:<6} : {:<5}".format('Label','Prediction'))
        print('------------------------------------------')
        for i in range(len(y_pred)):
            print ("{:.4f} : {:.4f}".format(self.y_trial[i],y_pred[i]))
        print('------------------------------------------')
        
        
    def model1(self,params, epochs = 5):
        hidL = params['hidL']
        hidU = params['hidU']
        hyperSet = params['lr']
        errTrain = []
        errTest = []
        self.TrainErr = []
        self.ValErr = []
        print('==> Model Selection......')
        for lr in hyperSet:
            start = time.time()
            for k in range(self.k):
                self.initialize(hidL, hidU)
                for epoch in range(epochs):
                    self.crossValidationSetGenerator(k, self.x_train, self.y_train)
                    self.forward(self.xtrain)
                    self.gradient(lr, self.ytrain)
                errTrain.append(self.modelEvaluate(self.xtrain, self.ytrain))
                errTest.append(self.modelEvaluate(self.xval, self.yval))
            self.TrainErr.append(np.mean(errTrain, axis = 0))
            self.ValErr.append(np.mean(errTest, axis = 0))
            errTrain = []
            errTest = []
            end = time.time()
            print('--> CV with hyperparameters [Alpha]:', lr)
            print('--> Time taken to complete above model training:{}'.format(end-start))
            print('\n')
        bestlr = hyperSet[np.argmin(np.array(self.ValErr)[:,0])]
        modellayout = 'Alpha* : ' + str(bestlr) 
        print('************************************************************************************')
        print('Best Learning Rate:', bestlr)
        print('************************************************************************************')
        print('\n')
        print('Model Selection is complete !!!!')
        print('\n')
        print('==> Model Retraining and Prediction.....')
        self.plotError(hyperSet,epochs, modellayout, retrain = False)
        self.predict([0, 0, bestlr],modellayout,hyperSet, epochs)
        self.evaluateTrial()
        print('==> Model Retraining and Predictions are complete!!!!')
        

    def model2(self,params, epochs = 5):
        hidL = params['hidL']
        hidU = params['hidU']
        learningRate = params['lr']
        self.TrainErr = {}
        self.ValErr = {}
        print('==> Model Selection......')
        for l in hidL:
            for u in hidU:
                for lr in learningRate:
                    start = time.time()
                    errTrain = []
                    errTest = []
                    for k in range(self.k):
                        self.initialize(l, [u]*l)
                        for epoch in range(epochs):
                            self.crossValidationSetGenerator(k, self.x_train, self.y_train)
                            self.forward(self.xtrain)
                            self.gradient(lr, self.ytrain)
                        errTrain.append(self.modelEvaluate(self.xtrain, self.ytrain))
                        errTest.append(self.modelEvaluate(self.xval, self.yval))
                    end = time.time()
                    key = str(l)+','+str(u)+','+str(lr)
                    print('--> CV with hyperparameters[Layers, Units, Alpha]:', key)
                    print('--> Time taken to complete above model training:{}'.format(end-start))
                    print('\n')
                    self.TrainErr[key]  = np.mean(errTrain, axis = 0)
                    self.ValErr[key] = np.mean(errTest, axis = 0) 
        TrainKeys = self.TrainErr.keys()
        keys = np.array([Keys.split(',') for Keys in TrainKeys], dtype = 'float32')
        hyperSet = list(self.TrainErr.keys())
        self.TrainErr = np.array(list(self.TrainErr.values()))
        self.ValErr = np.array(list(self.ValErr.values()))
        bestL, bestU, bestLR = keys[np.argmin(self.ValErr[:,0])]
        modellayout = 'Alpha* : ' + str(bestLR) + ' , ' + 'L* : ' + str(bestLR) +' , '+ 'U* : ' + str(bestU)
        print('************************************************************************************')
        print('Best Hyper Parameters: Hidden Layers - {}, Hidden Units - {}, Learning Rate - {}'.format(bestL, bestU, bestLR))
        print('************************************************************************************')
        print('\n')
        print('==> Model Retraining and Prediction.....')
        self.plotError(hyperSet,epochs, modellayout, retrain = False)
        self.predict([bestL, bestU, bestLR],modellayout,hyperSet, epochs)
        self.evaluateTrial()
        print('==> Model Retraining and Predictions are complete!!!!')
        

    def model3(self,params, epochs = 5):
        hidL = params['hidL']
        hidU = params['hidU']
        learningRate = params['lr']
        self.TrainErr = {}
        self.ValErr = {}
        print('==> Model Selection......')
        for u1 in hidU[0]:
            for u2 in hidU[1]:
                for lr in learningRate:
                    start = time.time()
                    errTrain = []
                    errTest = []
                    for k in range(self.k):
                        self.initialize(hidL, [u1,u2])
                        for epoch in range(epochs):
                            self.crossValidationSetGenerator(k, self.x_train, self.y_train)
                            self.forward(self.xtrain)
                            self.gradient(lr, self.ytrain)
                        errTrain.append(self.modelEvaluate(self.xtrain, self.ytrain))
                        errTest.append(self.modelEvaluate(self.xval, self.yval))
                    end = time.time()
                    key = str(u1)+','+str(u2)+','+str(lr)
                    print('--> CV with hyperparameters [HidU1, HidU2, Alpha]:', key)
                    print('--> Time taken to complete above model training:{}'.format(end-start))
                    print('\n')
                    self.TrainErr[key]  = np.mean(errTrain, axis = 0)
                    self.ValErr[key] = np.mean(errTest, axis = 0)          
        TrainKeys = self.TrainErr.keys()
        keys = np.array([Keys.split(',') for Keys in TrainKeys], dtype = 'float32')
        hyperSet = list(self.TrainErr.keys())
        self.TrainErr = np.array(list(self.TrainErr.values()))
        self.ValErr = np.array(list(self.ValErr.values()))
        bestU1, bestU2, bestLR = keys[np.argmin(self.ValErr[:,0])]
        modellayout = 'Alpha* : ' + str(bestLR) + ' , ' + 'U1* : ' + str(bestU1) +' , '+ 'U2* : ' + str(bestU2)
        print('************************************************************************************')
        print('Best Hyper Parameters: Hidden1 Units - {}, Hidden2 Units - {}, Learning Rate - {}'.format(bestU1, bestU2, bestLR))
        print('************************************************************************************')
        print('\n')
        print('==> Model Retraining and Prediction.....')
        self.plotError(hyperSet, epochs, modellayout, retrain = False)
        self.predict([2, [bestU1, bestU2], bestLR],modellayout,hyperSet, epochs)
        self.evaluateTrial()

#%% Model 1: Perceptron with no hidden layer
start = time.time()
model = NNet(1024,10)
model1Params = {'lr':[1, 4**1, 4**2, 4**3, 4**4],'hidL': 0 ,'hidU':None}

print('=========================================================================')
print(' Model Architecture: 1')
print('=========================================================================')
print('\n')
model.model1(model1Params, epochs = 1)

end = time.time()
print('\n Total Time Elapsed in training Model1:', end-start)
print('\n ==> Plotting Results !!!!')
#%% Model 2: Multi-Layer Perceptron with deep network
start = time.time()
model = NNet(1024,10)
model2Params = {'lr':[4**-2, 4**-1, 4**0, 4**1, 4**2],'hidL': [1,2,3,4] ,'hidU':[4**2,4**3,4**4]}

print('=========================================================================')
print(' Model Architecture: 2')
print('=========================================================================')
print('\n')
model.model2(model2Params, epochs = 1)

end = time.time()
print('\n Total Time Elapsed in training Model1:', end-start)
print('\n ==> Plotting Results !!!!')
#%% Model 3: Multi-Layer Perceptron with Two Hidden Layers
start = time.time()
model = NNet(1024,10)
model3Params = {'lr':[4**-3, 4**-2, 4**-1, 4**0, 4**1],'hidL': 2 ,'hidU':[[4**3,4**4],[4**2,4**3]]}

print('=========================================================================')
print(' Model Architecture: 3')
print('=========================================================================')
print('\n')
model.model3(model3Params, epochs = 1)

end = time.time()
print('\n Total Time Elapsed in training Model1:', end-start)
print('\n ==> Plotting Results !!!!')