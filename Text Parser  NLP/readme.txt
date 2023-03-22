***Text Mining and Information Retrieval (CIS-536)***
***Checkpoint - 01***

Author	: Nithesh Veerappa
Sudent ID	: 0188 3074
Date		: 03/05/2023

*Deliverables*
1. textparser.py	(Contains Source code defining the 'TextParser' Class for the given parsing task)
2. readme.txt	(Contains instruction to reun the code)
3. dictionary.txt	(Contains all the words(lemmatized) in the vocabulary arranged in alphabetical order)
4. unigrams.txt	(Contains List of words and their corresponding word_code, document frequency and global frequnecy)

*Librabries Used*
1. gzip (For unziping the tiny_wikipedia.txt.gz)
2. nltk (For lemmatizing the words in the vocabulary list)
3. re   (To parse the input text file using Regular Expressions)

*Instruction*
After placing textparser.py and tini_wikipedia.txt.qz in the same folder, follow below steps in the same sequence:
1. Installation of all the special libraries used in the source code
	$ !pip install nltk
	$ !pip install re
2. Downloading the wordnet to enable nltk library to perform NLP with respect to wordnet library.
   This is to be done only once.
	$ import nltk
	$ nltk.download('wordnet')
	$ nltk.download('omw-1.4')
3. Two ways to run the script
   a. Calling the script from command prompt:
	>> python textparser.py
   b. Running the script using an editor (VScode, spyder, vim etc)
	Run the script by clicking run 
4. After following the above steps, two files 'dictonary.txt' and 'unigrams.txt' will be created in the working directory.
   Status of every step will be printed in the command window to monitor the progress.

Unigram Computation is Complete !!
