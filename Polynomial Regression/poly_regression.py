# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:13:32 2023

@author: NitheshV

"""
import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide = 'ignore') 

lmbda = [0,np.exp(-25),np.exp(-20),np.exp(-14),np.exp(-7),np.exp(-3),1,np.exp(3),np.exp(7)]


class PolynomialRegression:
    def __init__(self, train_dat = 'train.dat', test_dat = 'test.dat', dim = 12, k = 6, lmbda = lmbda):
        self.train_data = open(train_dat, "r").readlines()
        self.test_data = open(test_dat, "r").readlines()
        self.dim = dim
        self.w = []
        self.lmbda = lmbda
        self.k = k
        self.ksize = int(len(self.train_data)/self.k)
        self.mean = [0]

        self.std = [1]
  
    
    def dataClean(self,data, xu = 0, xsig = 0, yu = 0, ysig = 0):
        x = []
        y = []
        for i in data:
            x.append(float(i.split(' ')[0]))
            y.append(float(i.split(' ')[1]))
        x, xu, xsig = self.normalization(x, u = xu, sig = xsig)
        y, yu, ysig = self.normalization(y, u = yu, sig = ysig)
        return (x, y),(xu,yu),(xsig, ysig)
    
    
    def normalization(self, data, u = 0, sig = 0):
        if u == 0 and sig == 0:
            mean = np.mean(data,axis=0)
            std = np.std(data,axis=0)
        else:
            mean = u
            std = sig
        normalized_data = []
        for i in data:
            normalized_data.append((i-mean)/std)
        return normalized_data, mean, std
    

    def inputTransfom(self, x,d):
        TinData = []
        TinData = np.ones_like(x)
        for i in range(1,d+1):
            nVec = np.power(x, i)
            if len(self.mean) <= i and len(self.mean) <= i:
                self.mean.append(np.mean(nVec))
                self.std.append(np.std(nVec))
                nVec = (nVec - self.mean[-1] )/self.std[-1]
            else:
                nVec = (nVec - self.mean[i] )/self.std[i]
            TinData = np.vstack((TinData, nVec))
        TinData = np.transpose(TinData)
        return TinData
    
    def cvScaling(self, d):
        for i in range(1,d+1):
            xmean = np.mean(self.xtrain[d])
            xstd = np.std(self.xtrain[d])
            self.xtrain[:,d] = (self.xtrain[:,d] - xmean)/xstd
            self.xval[:,d] = (self.xval[:,d] - xmean)/xstd
        
        ymean = np.mean(self.ytrain)
        ystd = np.std(self.ytrain)
        self.ytrain = (self.ytrain - ymean)/ystd
        self.yval = (self.yval - ymean)/ystd
                
                
    def rmseCal(self, y, yhat, ysig):
        return np.sqrt(np.mean(np.power((yhat-y)*ysig,2)))
    
    
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
            
    def modelEvaluation(self,xt,yt, ysig, w, d):
        xtest = self.inputTransfom(xt,d)
        ythat = np.dot(xtest,w)
        TestRmse = self.rmseCal(yt,ythat, ysig)
        print('--> Test Loss: {:0.4f}'.format(TestRmse))
        return TestRmse
        
    def printTable(self, train, val, ridge = False):
        if not ridge:
            facName = 'Dim'
            fac = np.arange(1,self.dim+1)
            
        else:
            facName = 'Lambda'
            fac = self.lmbda
        print('--> Traning and Validation losses Table:')
        print('------------------------------------------')
        print ("{:<23} {:<12}{:<10}".format(facName,'TrainLoss','ValLoss'))
        print('------------------------------------------')
        for i in range(len(train)):
            print ("{:<23} : {:.4f} : {:.4f}".format(fac[i],train[i],val[i]))
        print('------------------------------------------')
        
    def plotResults(self):
        (x,y),(xu,yu),(xsig,ysig) = self.dataClean(self.train_data)
        x_data = np.arange(1968,2024)
        
        # Polynomial Regression with LMSE
        x_data_nor = (x_data - xu)/xsig
        x_data_nor_PR = self.inputTransfom(x_data_nor,self.dPR)
        y_PR_nor = np.dot(x_data_nor_PR, self.wPR)
        y_PR = y_PR_nor*ysig + yu 
        
        # Polynomial Ridge Regression
        x_data_nor_PRR = self.inputTransfom(x_data_nor,self.dim)
        y_PRR_nor = np.dot(x_data_nor_PRR, self.wPRR)
        y_PRR = y_PRR_nor*ysig + yu
        
        #Original Data
        x_orig = np.array(x)*xsig + xu
        y_orig = np.array(y)*ysig + yu
        
        #Ploting Results
        plt.figure(dpi=100)
        plt.plot(x_data, y_PR, 'r', label = "Best d*")
        plt.plot(x_data, y_PRR, 'b--', label = "Best $\lambda$*")
        plt.scatter(x_orig, y_orig,color='green', label = "Origianl Data")
        plt.xlabel("Years")
        plt.ylabel("Age")
        plt.title("Curve fitting for optimal d and $\lambda$")
        plt.legend()
        plt.show()
    
    def polyFit(self,g):
        if g != 0:
            ridgeFac = g*np.eye(self.dim + 1)
            ridgeFac[0][0] = 0
            coMat = np.dot(np.linalg.inv(np.dot(np.transpose(self.xtrain), self.xtrain) + ridgeFac), np.transpose(self.xtrain))
        else:
            coMat = np.dot(np.linalg.inv(np.dot(np.transpose(self.xtrain), self.xtrain)), np.transpose(self.xtrain))
        return np.dot(coMat, self.ytrain)
    
    def polyRegression(self):
        self.ValRmse = []
        self.TrainRmse = []
        (x, y),(xu,yu),(xsig, ysig) = self.dataClean(self.train_data)
        (xt, yt),_,_ = self.dataClean(self.test_data, xu, xsig, yu, ysig)
        for d in range(1,self.dim+1):
            tx = self.inputTransfom(x,d)
            TrainRmse = 0
            ValRmse = 0
            for i in range(0, self.k):
                self.crossValidationSetGenerator(i,tx,y)
                #self.cvScaling(d)
                self.w = self.polyFit(g=0)
                self.yhat_train = np.dot(self.xtrain,self.w)
                self.yhat_val = np.dot(self.xval, self.w)
                TrainRmse += self.rmseCal(self.ytrain, self.yhat_train, ysig)
                ValRmse += self.rmseCal(self.yval, self.yhat_val, ysig)
            self.ValRmse.append(ValRmse/self.k)
            self.TrainRmse.append(TrainRmse/self.k)
        self.printTable(self.TrainRmse, self.ValRmse)
        plt.figure(dpi=100)
        plt.plot(np.arange(1,self.dim + 1), self.ValRmse,'r--', label = "Validation Loss")
        plt.plot(np.arange(1,self.dim + 1), self.TrainRmse,'y', label = "Training Loss")
        plt.legend()
        plt.xlabel('Polynomial Dimension')
        plt.ylabel('loss')
        plt.title('Training and Validation losses for Standard Polynomial Regression')
        
        #Retraining for optimal weights with best dimension obtained
        self.dPR = np.argmin(self.ValRmse)  + 1  
        self.xtrain = self.inputTransfom(x, self.dPR)
        self.ytrain = y
        self.wPR = self.polyFit(g=0)
        y_new = np.dot(self.xtrain,self.wPR)
        trainRMSE = self.rmseCal(self.ytrain, y_new, ysig)
        
        # Printing the training results obatined
        print('--> Best Dimension Obtained:', self.dPR)
        print('--> Polynomial Coefficients for best d:',self.wPR)
        print('--> Training Loss: {:0.4f}'.format(trainRMSE))
        
        #Model Evaluation
        self.modelEvaluation(xt, yt, ysig, w = self.wPR, d = self.dPR)


    def polyRidgeRegression(self):
        self.ValRmse = []
        self.TrainRmse = []
        self.WeightDict = {}
        #self.x, self.y = [],[]
        (x, y),(xu,yu),(xsig, ysig) = self.dataClean(self.train_data)
        (xt, yt),_,_ = self.dataClean(self.test_data, xu, xsig, yu, ysig)
        for g in self.lmbda:
            tx = self.inputTransfom(x,self.dim)
            TrainRmse = 0
            ValRmse = 0
            for i in range(0,self.k):
                self.crossValidationSetGenerator(i,tx,y)
                #self.cvScaling(self.dim)
                self.w = self.polyFit(g)
                self.yhat_train = np.dot(self.xtrain,self.w)
                self.yhat_val = np.dot(self.xval, self.w)
                TrainRmse += self.rmseCal(self.ytrain,self.yhat_train, ysig)
                ValRmse += self.rmseCal(self.yval,self.yhat_val, ysig)
            
            self.ValRmse.append(ValRmse/self.k)
            self.TrainRmse.append(TrainRmse/self.k)
        self.printTable(self.TrainRmse, self.ValRmse, ridge = True)    
        plt.figure(dpi=100)
        plt.xscale('log')
        plt.plot(self.lmbda, self.ValRmse, 'g--', label = "Validation Loss")
        plt.plot(self.lmbda, self.TrainRmse, 'r', label = "Training Loss")
        plt.legend()
        plt.xlabel('Lambda Coefficients')
        plt.ylabel('loss')
        plt.title('Training and Validation losses for Polynomial Ridge Regression')

        #Retraining for optimal weights with best lambda obtained
        bestG = np.argmin(self.ValRmse) 
        self.gPRR = self.lmbda[bestG] 
        self.xtrain = self.inputTransfom(x, self.dim)
        self.ytrain = y
        self.wPRR = self.polyFit(g= self.gPRR)
        y_new = np.dot(self.xtrain,self.wPRR)
        trainRMSE = self.rmseCal(self.ytrain, y_new, ysig)
        
        # Printing the training results
        print('--> Best Lambda Obtained: {:0.4f}'.format(self.gPRR))
        print('--> Polynomial Coefficients for best lambda:',self.wPRR)
        print('--> Training Loss: {:0.4f}'.format(trainRMSE))
        
        #Model Evaluation
        self.modelEvaluation(xt, yt, ysig, w = self.wPRR, d = self.dim)
#%%
# Instantiating the Class PolynomialRegression
pr = PolynomialRegression()
print('***Final Results***')
print('==========================================================')
print('Algo1: Curve fitting with Standard Polynomial Regression')
print('==========================================================')
pr.polyRegression()
print()
print('==========================================================')
print('Algo2: Curve fitting with Ridge Regression')
print('==========================================================')
pr.polyRidgeRegression()
print()
pr.plotResults()

        
                
            
              
                
                
                
            
            
        
        
    
    