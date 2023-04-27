# -*- coding: utf-8 -*-
"""
CIS 536 Text Mining and Information Retrieval

Project: Topic Modeling with Latent Dirichlet AllocationS

@author: Meeshawn Marathe (UMID: 4575 4188)
       : Nithesh Veerappa (UMID: 0188 3074)
"""
import os
import re
import nltk
import random
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stopwords.update(['said', 'u', 'could', 'want', 'would', 'get'])
stopwords.update(STOPWORDS)
from collections import Counter
import numpy as np
# np.random.seed(seed = 42)
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%

class TopicModeling:
  def __init__(self, dataset, num_topics):
    self.lemObj = WordNetLemmatizer()
    self.lemmTable = {}
    self.dataParse(dataset) 
    self.k = num_topics
    self.alpha = 0.1
    self.beta = 0.5
    
  def dataClean(self, doc):
    filePointer = open(doc,'rt')
    doc = filePointer.read()
    filePointer.close()
    
    # Remove urls/unwanted characters
    corpus = re.sub('http[^\s]+','',doc)
    corpus = re.findall(r'\b[\w]+\b',corpus.lower())  
    corpus = [word for word in corpus if word not in stopwords]
    
    # Perform Lemmatization
    for word in set(corpus):
        self.lemmTable[word] = self.lemObj.lemmatize(word) 

    return [self.lemmTable[word] for word in corpus]
    
    
  def dataParse(self, dataset):
    doc_id = 0
    self.list_of_docs = {}
    for folder in os.listdir(dataset):
      for file in os.listdir(dataset + '/' + folder):
        file = dataset + '/' + folder +'/' + file
        self.list_of_docs[doc_id] = self.dataClean(file)
        doc_id += 1
   
  
  def intialization(self):
    self.topics_per_doc = {}
    self.topics_per_word = {}
    self.vocab = set()
    
    for doc_id, doc in self.list_of_docs.items():
      wrd_topic_list = []
      for word in doc:
        self.vocab.add(word)
        word = word + ' ' + 'topic' + str(np.random.choice(np.arange(self.k)))
        wrd_topic_list.append(word)
      self.list_of_docs[doc_id] = wrd_topic_list
      
      # theta_d_z[d][z]
      self.topics_per_doc[doc_id] = Counter([pair.split()[1] for pair in wrd_topic_list])
    print("Initializaton completed!")
    
    dummy = []
    list(dummy.extend(lis) for lis in self.list_of_docs.values())
    self.topics_per_word = Counter(dummy)  
    
    self.topics = {}
    for doc, val in self.topics_per_doc.items():
      for topic, count in val.items():
        self.topics[topic] = self.topics.get(topic, 0) + count
    

  def randomWalk(self, iters):
    print("\n Random Walk initiated")
    for _ in tqdm(range(iters)):
      for doc_id, doc in self.list_of_docs.items():
        for wrd_id, word_and_topic in enumerate(doc):
          #Fetch the topic from the word
          word, topic = word_and_topic.split()
    
          # Decrement counts for the associated topic for the word in doc
          self.topics_per_doc[doc_id][topic] -= 1
          self.topics_per_word[word + " " + topic] -= 1
          self.topics[topic] -= 1
          
          # Sampling a new topic from multinomial distribution
          dist_topics_per_doc = (np.array(list(self.topics_per_doc[doc_id].values())) + self.alpha)/\
                                (len(doc) - 1 + self.k*self.alpha)
          num_a = np.array([self.topics_per_word[word + ' ' + topic] for topic in ['topic' + str(i) for i in range(self.k)]])                        
          dist_words_per_topic = (num_a + self.beta)/(len(self.topics) + self.beta*len(self.vocab))
          
          new_dist_topic = dist_topics_per_doc*dist_words_per_topic
          new_dist_topic = new_dist_topic/np.sum(new_dist_topic)
          # new_topic = 'topic' + str(np.random.multinomial(1, new_dist_topic).argmax())
          new_topic = 'topic' + str(random.choices(np.arange(self.k),weights = new_dist_topic, k=1)[0])
          # Assign the new topic to the original word
          
          self.list_of_docs[doc_id][wrd_id] = word + ' ' + new_topic
          self.topics_per_doc[doc_id][new_topic] += 1
          self.topics_per_word[word + " " + new_topic] += 1
          self.topics[new_topic] += 1
                                
  def plotTopicDistForDoc(self, doc_id):
      doc_topics = list(self.topics_per_doc[doc_id].values())
      sum = np.sum(doc_topics)
      y = [y/sum for y in doc_topics]
      plt.bar(range(len(doc_topics)),y)
    
    
  def plotWordDistForTopic(self):
      topics = ["topic" + str(i) for i in range(self.k)]
      for topic in topics:
          words_and_counts = [np.array([word_and_topic.split()[0], counts]) for word_and_topic, counts in self.topics_per_word.items() if word_and_topic.split()[1] == topic]
          words_and_counts = np.array(sorted(words_and_counts, key = lambda x:int(x[1]), reverse=True)[:20])        
          fig, ax = plt.subplots()
          ax.bar(words_and_counts[:,0], np.array(words_and_counts[:,1], dtype='int'))
          ax.set_xticklabels(words_and_counts[:,0], rotation=90)
          ax.set_title("Topic Distribution over top-20 words ({})" .format(topic))
                                        
  
#%%
dataset = 'datasetTopicModeling'
LDA = TopicModeling(dataset, num_topics=5)
LDA.intialization()
LDA.randomWalk(iters=200)
LDA.plotTopicDistForDoc(doc_id = 1)
LDA.list_of_docs[3]
LDA.plotWordDistForTopic()