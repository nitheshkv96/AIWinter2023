# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:46:25 2023

@author: NitheshV
"""
import gzip
from tqdm import tqdm
import re

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

# from functools import lru_cache
#%%
class TextParser():
    def __init__(self, file ='tiny_wikipedia.txt.gz'):
        self.file = file
        self.wiki = None
        self.GlobFreq = {}
        self.DocFreq = {}
        self.word2idx = {}
        self.lemmatizer = WordNetLemmatizer()
        self.loadData()
        self.vocab = self.clean(self.wiki)
        self.lemmaDict = self.lemmatize(set(self.vocab))
        
        
    def loadData(self):
        '''
        This function unzips the zipped document and stores the content
        of the document in a variable
        '''
        with gzip.open(self.file, 'rt') as f:
            self.wiki = f.read()
            
            
    def clean(self,line):
        '''
        This function tokenize every line in the document 
        and discards all special characters and unwanted texts in the document 
        e.g. url, strings with special charaters, punctuations etc
        '''
        col = re.sub(r'http\S+', '', line) # to remove urls
        col = re.sub(r'[#,$,@,%,&,*,]\S+','', col) # remove strings with special characters
        vocabAlp = re.findall('[A-Za-z]\w+', col.lower()) # Tokenization of all strings purely aplhabetical
        vocabNum = re.findall('[0-9]\w+', col) # Tokenization of all strings purely numerical
        return vocabAlp + vocabNum
    

    def lemmatize(self, vocab):
        '''
        This function lemmatize all tokens in the vocabulary list
        '''
        lemmDic = {}
        print('==> Lemmatization of all tokens in vocabulary:')
        for token in tqdm(vocab):
            lemmDic[token] = self.lemmatizer.lemmatize(token)
        return lemmDic
    
    # @lru_cache(maxsize = 1000)
    def tokenCount(self):
        '''
        This function create dictonaries for
        Global Frequency: Number of times every token throughout the document
        Document Frequency: Number of documents every token occur in
        '''
        print('==> Computation of Global frequency and Document Frequency for all the words in vocabulary:')
        for token in tqdm(self.vocab):
            token = self.lemmaDict[token]
            if token not in self.GlobFreq:
                self.GlobFreq[token] = 1
            else:
                self.GlobFreq[token] += 1
        for line in tqdm(self.wiki.splitlines()):
            for token in set(self.clean(line)):
                token = self.lemmaDict[token]
                if token not in self.DocFreq:
                    self.DocFreq[token] = 1
                else:
                    self.DocFreq[token] += 1
        

    def textWrite(self):
        '''
        This function sorts the dictonary and generate two files
        unigrams.txt : <word_code> <token> <doc freq> <glob freq reverse sorted> 
        dictonary.txt: <token sorted alphabetically>
        '''
        f1 = open('unigrams.txt', 'w')
        f2 = open('dictionary.txt', 'w')
        c = 0
        print('==> Writing the computed Global frequency and Document frequency into a text file:')
        print('.')
        print('.')
        print('.')
        for key in sorted(self.GlobFreq.keys()):
            self.word2idx[key] = c
            f2.write(key + '\n')
            c+= 1
        self.GlobFreq = dict(sorted(self.GlobFreq.items(), key=lambda item: item[1], reverse = True))
        for key, val in self.GlobFreq.items():
            f1.write(str(self.word2idx[key])+ ' '+ '{:20s}'.format(key) + ' '+ str(self.DocFreq[key])  + ' '+  str(val) + '\n')
            
        print('Text files created, Unigram Computation is Complete !!')
        f1.close()
        f2.close()
        
#%% Instantiating TextParser class and calling its methods to compute tokens and their frequencies
# import time

# # get the start time
# st = time.time()
tp = TextParser()
tp.tokenCount()
tp.textWrite()
# get the end time
# et = time.time()

# # get the execution time
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')