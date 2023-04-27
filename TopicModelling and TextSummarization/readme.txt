***Text Mining and Information Retrieval***
***Project***

***Topics***
1. Text summarization using page-rank algorithm
2. Topic Modeling using Gibs-Sampling based Latent Dirichlet Allocation

Author	: Nithesh Veerappa
Date		: 04/18/2023

*Deliverables*
1. ldaTopicModeling.py	 (Contains Source code for Topic Modeling Task)
2. pageRank.py		 (Contains Source code for Text Summarization Task)
3. readme.txt	 	 (Contains instruction to execute the code)
4. Dataset           (Contains BBC news articles and CNN daily mail News for two tasks is uploded in Google drive. Link is provided below)


*Librabries Used*
1. os (For reading the input text files)
2. nltk (For lemmatizing the words in the vocabulary list and stopwords)
3. gensim (For stop words)
4. re   (To parse the input text file using Regular Expressions)
5. collections (Counter is used to get the term frequency)
6. numpy (For matrix and math operations)
7. random (For random number generations)
8. matplotlib (For plotting results)

*Instruction*
After placing "ldaTopicModeling.py", "pageRank.py" and folder "dataset" (please unzip and delete the zip file) containing input dataset in the same directory.
Two datasets one for each task are zipped together. After extraction please make sure "datasetTopicModeling" folder and "ldaTopicModeling.py" are placed together and "datasetPageRank" folder and "pageRank.py" are placed together in their respective directory when executing the code.
Follow below steps in the same sequence:
1. Installation of all the special libraries used in the source code (if not installed already!!)
	$ !pip install nltk
	$ !pip install re
	$ !pip install gensim
	$ !pip install matplotlib
2. Downloading the wordnet to enable nltk library to perform NLP with respect to wordnet library.
   This is to be done only once.
	$ import nltk
	$ nltk.download('wordnet')
	$ nltk.download('omw-1.4')
3. Two ways to run the script (call one after the other)
   a. Calling the script from command prompt:
	>> python pagerank.py
	>> python ldaTopicModeling.py
   b. Running the script using an editor (VScode, spyder, vim etc)
	Run the script by clicking run 
4. After following the above steps, results will be plotted in the same environment where the scripts are called in.

Note:
The google drive link for the dataset used in this project can be found below:
https://drive.google.com/drive/folders/17C6Ru0IzpaTWrQfcSJfqoxcfc6pM0qoA?usp=share_link


End of Readme !!
