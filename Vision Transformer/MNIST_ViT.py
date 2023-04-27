# -*- coding: utf-8 -*-
"""
CIS 590K Deep Learning (WINTER 2023)
@author: Meeshawn Marathe (4575 4188)
         Nithesh Veerappa (UMID: 0188 3074)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

#%%
class ViT(nn.Module):
  # def __init__(self, L, HUratio, n_blocks, origImgDim=(1,28,28), subImageDim=7, hidden_d=8, n_heads=2, out_d=10):
  def __init__(self, HUratio, n_blocks, origImgDim=(1,28,28), subImageDim=7, hidden_d=8, n_heads=2, out_d=10):    
    super(ViT, self).__init__()
    self.origImgDim = origImgDim
    self.subImageDim = subImageDim
    
    self.patch_size = (self.origImgDim[1] / self.subImageDim, self.origImgDim[2] / self.subImageDim)
    
    # Mapping to a lower dimensional embedding
    input_d = int(self.origImgDim[0] * self.patch_size[0] * self.patch_size[1])
    self.embedding = nn.Linear(input_d, hidden_d)
    
    # Learnable classifiation token
    self.class_token = nn.Parameter(torch.rand(1, hidden_d))
    
    # Positional embedding
    posEmbedding = torch.ones(self.subImageDim**2 + 1, hidden_d)
    for i in range(posEmbedding.shape[0]):
        for j in range(hidden_d):
            posEmbedding[i][j] = np.sin(i / (10000 ** (j / hidden_d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / hidden_d)))
    
    # posEmbedding = nn.Parameter(torch.tensor(posEmbedding))
    # posEmbedding.requires_grad = False
    self.register_buffer('posEmbedding', posEmbedding, persistent=False)
    
    # Create additional Encoder blocks
    # self.blocks = nn.ModuleList([ViTBlock(hidden_d, HUratio, n_heads, L=L) for _ in range(n_blocks)])
    self.blocks = nn.ModuleList([ViTBlock(hidden_d, HUratio, n_heads) for _ in range(n_blocks)])
    
    # Create a dense network for the classification task
    self.mlp = nn.Sequential(nn.Linear(hidden_d, out_d),nn.Softmax(dim=-1))
    
    
  def createSubImages(self, images):
    m, channel, height, width = images.shape

    assert height == width, "createSubImages method is implemented for square images only"

    subImages = torch.zeros(m, self.subImageDim ** 2, height * width * channel // self.subImageDim ** 2)
    subImgSize = height // self.subImageDim # 4

    for idx, image in enumerate(images):
        for i in range(self.subImageDim):
            for j in range(self.subImageDim):
                subImage = image[:, i * subImgSize: (i + 1) * subImgSize, j * subImgSize: (j + 1) * subImgSize]
                subImages[idx, i * self.subImageDim + j] = subImage.flatten()
    return subImages

  def forward(self, images):
    m, channel, height, width = images.shape
    subImages = self.createSubImages(images).to(self.posEmbedding.device)
    subImages = self.embedding(subImages)
    # Appending classification embedding to along with the sub-image embeddings 
    # for every image
    subImages = torch.cat((self.class_token.expand(m, 1, -1), subImages), dim=1)
    
    # Adding positional embeddings
    out = subImages + self.posEmbedding.repeat(m, 1, 1)
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)
        
     # Extracting the classification token only
    out = out[:, 0]
    
    # Return the classification prediction
    return self.mlp(out)
 
#%%
class ViTBlock(nn.Module):
    # def __init__(self, hidden_d, HUratio, n_heads, L):
    def __init__(self, hidden_d, HUratio, n_heads):
        super(ViTBlock, self).__init__()        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        
        # modules = []
        # modules.append(nn.Linear(hidden_d, mlp_ratio * hidden_d))
        # modules.append(nn.GELU())

        # for _ in range(L-1):
        #   modules.append(nn.Linear(mlp_ratio * hidden_d, mlp_ratio * hidden_d))
        #   modules.append(nn.GELU())
        
        # modules.append(nn.Linear(mlp_ratio * hidden_d, hidden_d))
        # modules.append(nn.GELU())
        # self.mlp = nn.Sequential(*modules)        
        
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, HUratio * hidden_d),
            nn.GELU(),
            nn.Linear(HUratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out 
 
#%%
class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        d_head = int(d / n_heads) # 4
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head # 4
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])  
   
#%%  
class MNISTViT:
  def __init__(self, num_epochs):
    self.num_epochs = num_epochs
    self.nFolds = 3
    # self.alphaModelSelection = [0.001, 0.003, 0.01, 0.03]
    # self.numEncoderBlocks = [1, 2, 3]
    # self.numHiddenLayersMLP = [1, 2]
    self.alphaModelSelection = [0.005]
    self.numEncoderBlocks = [2,3]
    self.numHiddenUnitsRatio = [2,4]
    self.bestModelParams = None
    
    self.trainOrig = MNIST(root='./datasets', train=True, download=True, transform=ToTensor())
    self.testOrig = MNIST(root='./datasets', train=False, download=True, transform=ToTensor())
    # self.train = DataLoader(train, shuffle=True, batch_size=128)
    # self.test = DataLoader(test, shuffle=False, batch_size=128)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", self.device, f"({torch.cuda.get_device_name(self.device)})" if torch.cuda.is_available() else "")

  def modelSelection(self):
    # 3-fold cross validation across all the possible models based on 
    # various hyperparameters:
    self.trainMisClassifyLoss = {}
    self.testMisClassifyLoss = {} 
    splits = KFold(n_splits=self.nFolds,shuffle=False)
    
    for self.n_blocks in self.numEncoderBlocks:
      # for self.L in self.numHiddenLayersMLP:
      for self.HUratio in self.numHiddenUnitsRatio:
        for self.alpha in self.alphaModelSelection:
          # define losses here
          trainMisClassifyLoss = 0
          testMisClassifyLoss = 0
          for fold, (train_idx, valid_idx) in enumerate(splits.split(self.trainOrig)):
            # Prepare data
            train = SubsetRandomSampler(train_idx)
            valid = SubsetRandomSampler(valid_idx)
            self.train = DataLoader(self.trainOrig, batch_size=128, sampler=train)
            self.test = DataLoader(self.trainOrig, batch_size=128, sampler=valid)
            print("===============================================")
            # print("# Enc blocks = {}, MLP L1 HU = {}, alpha={}, fold-{}" .format(self.n_blocks,self.L,self.alpha, fold+1))            
            # self.compileModel(self.L, self.HUratio)
            self.compileModel()
            print("# Enc blocks = {}, MLP L1 HU = {}, alpha={}, fold-{}" .format(self.n_blocks,self.HUratio*8,self.alpha, fold+1))            

            self.fit()
            self.computeLoss()
            trainMisClassifyLoss += self.trainMisClassifyErr
            testMisClassifyLoss += self.testMisClassifyErr
            
          # idx = "Enc_blocks=" + str(self.n_blocks) + " L=" + str(self.L) + " a=" + str(self.alpha)
          idx = "Enc_blocks=" + str(self.n_blocks) + " HU=" + str(self.HUratio*8) + " a=" + str(self.alpha)
          self.trainMisClassifyLoss[idx] = trainMisClassifyLoss/self.nFolds
          self.testMisClassifyLoss[idx] = testMisClassifyLoss/self.nFolds
          
    bestModel = min(self.testMisClassifyLoss, key=self.testMisClassifyLoss.get)  
    self.bestModelParams = np.array([val.split('=') for val in bestModel.split()])[:,1].astype('double')          

  def plotHistory(self):
    fig, ax = plt.subplots()
    fig.dpi = 300
    trainMisClassifyLoss = list(self.trainMisClassifyLoss.values())
    testMisClassifyLoss = list(self.testMisClassifyLoss.values())
    x = list(self.trainMisClassifyLoss)
    ax.plot(x, trainMisClassifyLoss, 'o-', label="Training")
    ax.plot(x, testMisClassifyLoss, 'o-', label="Validation")
    ax.set_xlabel("Hyperparameters [# Enc blocks, MLP L1 HU, $\\alpha$]", fontweight='bold')
    ax.set_xticklabels(x, rotation=90, fontsize=6)
    ax.set_ylabel("Misclassification Loss", fontweight='bold')
    ax.set_title("Model Selection ({}-fold CV, {} epochs each: \n(Best # Enc blocks*={}, MLP L1 HU*={}, $\\alpha$*= {})" .format(self.nFolds, self.num_epochs, self.bestModelParams[0], self.bestModelParams[1], self.bestModelParams[2]),fontweight='bold')
    ax.legend(loc='lower right')
    ax.plot(np.argmin(np.array(testMisClassifyLoss)), min(testMisClassifyLoss), marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
    fig.tight_layout() 
    
    
  # def compileModel(self, L):
  def compileModel(self):
    shape = self.train.dataset.train_data.shape
    # self.model = ViT(L=L, HUratio=self.HUratio, n_blocks=self.n_blocks, origImgDim=(1, shape[1], shape[2]), subImageDim=7, hidden_d=8, n_heads=2, out_d=10).to(self.device)    
    self.model = ViT(HUratio=self.HUratio, n_blocks=self.n_blocks, origImgDim=(1, shape[1], shape[2]), subImageDim=7, hidden_d=8, n_heads=2, out_d=10).to(self.device)    

  def fit(self, cv=True):
    optimizer = Adam(self.model.parameters(), lr=self.alpha)
    self.crossEntropyLoss = CrossEntropyLoss()
    self.trainProxyErr = 0.0
    self.trainMisClassifyErr = 0.0

    self.trainProxyErr_per_epoch = []
    self.trainMisClassifyErr_per_epoch = []
    self.testProxyErr_per_epoch = []
    self.testMisClassifyErr_per_epoch = []    
    for epoch in trange(self.num_epochs, desc="Train"):
        train_loss = 0.0
        inCorrect, total = 0, 0

        for batch in tqdm(self.train, desc=f"Epoch {epoch + 1} in training", position = 0,leave=True):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat =self.model(x)
            loss = self.crossEntropyLoss(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(self.train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            inCorrect += torch.sum(torch.argmax(y_hat, dim=1) != y).detach().cpu().item()
            total += len(x)
            
        self.trainProxyErr_per_epoch.append(train_loss)
        self.trainMisClassifyErr_per_epoch.append(inCorrect/total)
        if not cv:
          self.computeLoss()
          self.testProxyErr_per_epoch.append(self.testProxyErr)
          self.testMisClassifyErr_per_epoch.append(self.testMisClassifyErr)
                    
        print(f"Epoch {epoch + 1}/{self.num_epochs} Proxy Train loss: {train_loss:.2f}")
        print(f"Epoch {epoch + 1}/{self.num_epochs} Misclassification Train loss: {inCorrect/total:.2f}")
    self.trainProxyErr = train_loss
    self.trainMisClassifyErr = inCorrect / total 
    
        
  def computeLoss(self):
    with torch.no_grad():
       inCorrect, total = 0, 0
       test_loss = 0.0

       for batch in tqdm(self.test, desc="Testing",position = 0,leave=True):
           x, y = batch
           x, y = x.to(self.device), y.to(self.device)
           y_hat = self.model(x)
           loss = self.crossEntropyLoss(y_hat, y)
           test_loss += loss.detach().cpu().item() / len(self.test)

           inCorrect += torch.sum(torch.argmax(y_hat, dim=1) != y).detach().cpu().item()
           total += len(x)
           
       self.testProxyErr = test_loss
       print(f"Proxy Test loss: {test_loss:.2f}")
       self.testMisClassifyErr = inCorrect / total
       print(f"Misclassification Test Loss: {self.testMisClassifyErr:.2f}")
       
  def evaluate(self):
    self.train = DataLoader(self.trainOrig, shuffle=True, batch_size=128)
    self.test = DataLoader(self.testOrig, shuffle=False, batch_size=128)
    self.n_blocks = int(self.bestModelParams[0])
    self.HUratio = int(self.bestModelParams[1]/8) 
    self.alpha = self.bestModelParams[2]
    self.compileModel()
    print("=========================================================")
    print("Training on the best model parameters:")
    print("# Enc blocks = {}, MLP L1 HU = {}, alpha={}" .format(self.n_blocks,self.HUratio*8,self.alpha))            
    self.fit(cv=False)
    
    # Plot the training results for the best model over the # of training iterations
    fig, ax = plt.subplots()
    fig.dpi = 300     
    ax.plot(np.arange(1,self.num_epochs+1), self.trainMisClassifyErr_per_epoch, 'o-', label="Training")
    ax.plot(np.arange(1,self.num_epochs+1), self.testMisClassifyErr_per_epoch, 'o-', label="Test")
    ax.set_xlabel("Epochs", fontweight='bold')
    ax.set_ylabel("Misclassification Loss", fontweight='bold')
    ax.set_title("Best ViT Model Fit over {} epochs \n(# Enc blocks* = {}, MLP L1 HU* = {}, $\\alpha$* = {}): " .format(self.num_epochs, self.n_blocks, self.HUratio, self.alpha), fontweight='bold')
    ax.legend(loc='upper right')
    fig.tight_layout() 
    
    fig2, ax2 = plt.subplots()
    fig2.dpi = 300
    ax2.plot(np.arange(1,self.num_epochs+1), self.trainProxyErr_per_epoch, 'o-', label="Training")
    ax2.plot(np.arange(1,self.num_epochs+1), self.testProxyErr_per_epoch, 'o-', label="Test")
    ax2.set_xlabel("Epochs", fontweight='bold')
    ax2.set_ylabel("Proxy Loss", fontweight='bold')
    ax2.set_title("Best ViT Model Fit over {} epochs \n(# Enc blocks* = {}, MLP L1 HU* = {}, $\\alpha$* = {}): " .format(self.num_epochs, self.n_blocks, self.HUratio, self.alpha), fontweight='bold')
    ax2.legend(loc='upper right')
    fig2.tight_layout() 

#%%
classifier = MNISTViT(num_epochs=5)
classifier.modelSelection()
classifier.plotHistory()
classifier.evaluate()
plt.show()

