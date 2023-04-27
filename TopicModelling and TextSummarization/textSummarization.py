# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:32:33 2023

@author: vicky
"""
"""
"TEXT SUMMARIZATION USING GRAPH-BASED APPROACH (PAGE-RANK ALGORITHM)"
@author: Meeshawn Marathe
"""
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np
import matplotlib.pyplot as plt

# NOTE: nltk library is only used for sentence segmentation.
# The Page-Rank algorithm has been implemented from scratch in
# python.

#%% Class Definition for Page-Rank based Text Summarization

class TextSummPageRank:
    
    def __init__(self, dataset, num_datapoints):
        self.dataset = dataset
        self.num_datapoints = num_datapoints
        self.x_list_of_articles = []
        self.y_list_of_highlights = []
        self.y_hat_list_of_highlights = []
        self.punc_list = [',', '.', ';', ':', '\'', '\"' , '-', ')', '(', '[', ']', '!', '?']
        self.damping = 0.85
        self.threshold = 0.0001
        self.shrink_factor = 0.15
        self.list_of_weighted_scores = []
        self.list_num_of_iterations = []
        self.list_rouge_score = []
            

    def dataClean(self, word):        
        # Remove punctuations at the begining and end of a word
        if word[-1] in self.punc_list:
            word = word[:-1]
        if word[0] in self.punc_list:
            word = word[1:]
        return word
    
    
    def dataParse(self):
        print('\nParsing through the data to extract and store {} new articles along with their annotated highlights/summary...' .format(self.num_datapoints))
        dataFrame = pd.read_csv(self.dataset, nrows = self.num_datapoints)
        # print(dataFrame.head())
        
        list_articles = dataFrame['article']             
        for article in list_articles:
            # article = ' '.join(sent_tokenize(article))
            article = sent_tokenize(article)
            article = [sentence.lower() for sentence in article]
            modified_article = []
            for sentence in article:
                sentence = [word for word in sentence.split(' ') if len(word)!= 0 and (len(word) != 1 or word not in self.punc_list)]
                sentence = [self.dataClean(word) for word in sentence]
                sentence = ' '.join(sentence) 
                modified_article.append(sentence)
            self.x_list_of_articles.append(modified_article)
            
        list_highlights = dataFrame['highlights']    
        for highlight in list_highlights:
            highlight = ' '.join(sent_tokenize(highlight))
            highlight = highlight.lower()
            highlight = [word for word in highlight.split(' ') if len(word)!= 0 and (len(word) != 1 or word not in self.punc_list)]
            highlight = [self.dataClean(word) for word in highlight]
            highlight = ' '.join(highlight)
            self.y_list_of_highlights.append(highlight)
        
        print('Done')
            
    def computeSentSimScore(self, sent1, sent2):
        overlap = [word for word in sent1.split(' ') if word in sent2.split(' ')]
        len_sent1 = len(sent1.split(' '))
        len_sent2 = len(sent2.split(' '))
        
        if len_sent1 == 1:
            log_sent_1 = 1
        else:
            log_sent_1 = np.log2(len_sent1)
            
        if len_sent2 == 1:
            log_sent_2 = 1
        else:
            log_sent_2 = np.log2(len_sent2)
            
        score = len(overlap)/(log_sent_1 + log_sent_2)
        return score
                   
                
    def pageRankAlgo(self, weighted_scores, sentSimScore):
        N = len(weighted_scores)
        D = self.damping
        new_weighted_score = []
        
        for node in range(N):# Iterating through all the sentence nodes
            indices = []
            # Computing a sentence's neighbor
            for neighbor in range(N):
                if neighbor != node:
                    indices.append(str(node) + ' ' + str(neighbor))
            
            # Computing over the sentence's neighbors
            sum_neighbors = 0
            for index in indices:
                neighbor = int(index.split(' ')[1])
                
                if sentSimScore.get(index) == None:
                    index = index.split(' ')
                    index = index[1] + ' ' + index[0]

                num = sentSimScore[index]*weighted_scores[neighbor]
               
                neighbors_of_neighbor_sum = 0
                for n in range(N):
                    if n != neighbor:
                        idx_neighbors_of_neighbor = str(neighbor) + ' ' + str(n)
                        if sentSimScore.get(idx_neighbors_of_neighbor) == None:
                            idx_neighbors_of_neighbor = idx_neighbors_of_neighbor.split(' ')
                            idx_neighbors_of_neighbor = idx_neighbors_of_neighbor[1] + ' ' + idx_neighbors_of_neighbor[0]
                        neighbors_of_neighbor_sum += sentSimScore[idx_neighbors_of_neighbor]
                
                den = neighbors_of_neighbor_sum
                # print(den)
                if den!= 0.0:
                    sum_neighbors += num/den
                else:
                    sum_neighbors = 0
                    
            new_ws = (1-D)/N + D*sum_neighbors
            new_weighted_score.append(new_ws)
        
        return new_weighted_score
    

    def computePageRank(self):
        print('\nComputing the weighted scores using PageRank algorithm for every sentence in an article across all the articles...')
        for article in self.x_list_of_articles:
            # STEP 1: Compute all the sentence similarity scores within
            # an article.
            sentSimScore = {}
            for i in range(len(article)):
                for j in range(i+1, len(article)):
                    indices = str(i) + ' ' + str(j)
                    sentSimScore[indices] = self.computeSentSimScore(article[i],article[j])
                    
            # STEP 2: Initialize the weighted score of each node with the same
            # value.
            weighted_scores = [0.25]*len(article)
            
            # STEP 3: Repeatedly calculate the weighted scores until the 
            # difference between the current score and its predecessor
            # is less than or equal to a threshold value
            num_iter = 0           
            while(True):
                new_weighted_scores = self.pageRankAlgo(weighted_scores, sentSimScore)
                num_iter = num_iter + 1
                delta = abs(np.array(weighted_scores)-np.array(new_weighted_scores))
                withinThresh = [val<=self.threshold for val in delta]
                weighted_scores = new_weighted_scores
                if all(withinThresh):
                    break;
                    
            self.list_num_of_iterations.append(num_iter)
            self.list_of_weighted_scores.append(new_weighted_scores)
            
        print('Done')
            
        
    def summarize(self):
        print('\nCreating a summary for each article by ranking the sentence based on their computed Page-Rank scores and concatenating them...')
        idx_article = 0
        for weighted_scores in self.list_of_weighted_scores:
            num_of_summ_lines = round(self.shrink_factor*len(weighted_scores))
            ws = np.array(weighted_scores)
            idx_top_N_summary = ws.argsort()[-num_of_summ_lines:][::-1]
            top_N_summary = ''
            for idx in idx_top_N_summary:
                top_N_summary = top_N_summary + ' ' + self.x_list_of_articles[idx_article][idx]
            self.y_hat_list_of_highlights.append(top_N_summary)
            idx_article = idx_article + 1
        print('Done')
            
    def computeRougeUnigram(self):
        print('\nEvaluating the performance metrics by computing Rouge-1 score for every article...')
        for y, y_hat in zip(self.y_list_of_highlights, self.y_hat_list_of_highlights):
            overlap = [word for word in y.split(' ') if word in y_hat.split(' ')]
            len_y = len(y.split(' '))
            rougeScore = len(overlap)/(len_y)
            self.list_rouge_score.append(rougeScore)
         
        sum_rouge1 = np.sum(np.array(self.list_rouge_score))
        average_rouge = sum_rouge1 / len(self.list_rouge_score)
        print('\nAverage Rouge-1 score of {} articles: {:0.2f}' .format(self.num_datapoints,average_rouge))
        
    def writeSummary(self):
        out_file = 'list_of_summary.txt'
        print('\nWriting data to {} ...' .format(out_file))
        file = open(out_file, 'w', encoding="utf-8")

        file.write('Page-Rank Generated Summaries\n')
        
        idx = 1
        for y_hat in self.y_hat_list_of_highlights:
            file.write('[' + str(idx) + ']' + ' ' + y_hat + "\n")
            idx = idx + 1
        
        file.write('\n')
        file.close()
        print('Done')
        
    def printRougeHistogram(self):
        fig, ax = plt.subplots()
        ax.hist(self.list_rouge_score, bins = 20)
        plt.xlabel('ROUGE-1 Scores')
        plt.ylabel('Frequency')
        plt.title('Frequency Histogram for PageRank Algorithm on {} articles' .format(self.num_datapoints))
        plt.show()
        
    def printNumIterations(self):
        plt.plot(self.list_num_of_iterations)
        plt.ylabel('Number of iterations')
        plt.xlabel('{} articles across the corpus' .format(self.num_datapoints))
        plt.title('Number of iterations made to converge by each of the {} articles' .format(self.num_datapoints))
        plt.show()
        
#%%                       
def main():
    # Initializing dataset
    dataset = './dataset/train.csv'

    # Setting the number of datapoints to process at a time
    num_datapoints = 500
     
    # Creating an instance of the TextSummPageRank class
    textSumm = TextSummPageRank(dataset, num_datapoints)
    
    # Parsing through the data to extract and store new articles along with
    # their annotated highlights/summary
    textSumm.dataParse()
    
    # Computing the weighted scores for each sentence in an article across 
    # all the articles in the corpus using the Page-Rank algorithm
    textSumm.computePageRank()
    
    # Creating a summary for each article by ranking the sentence based on 
    # their computed Page-Rank scores and concatenating them to provide a
    # summary
    textSumm.summarize()
    
    # Evaluating the performance metrics by computing Rouge-1 score for every
    # article
    textSumm.computeRougeUnigram()
    
    # Writing Predicted Summaries to a text file
    textSumm.writeSummary()
    
    # Plotting the Rouge-1 score histogram
    textSumm.printRougeHistogram()
    
    # Plotting the number of iterations made by each article to converge
    textSumm.printNumIterations()
    
if __name__ == "__main__":
    main()