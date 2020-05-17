### This repository contains all the relevant information and code for predicting the sentiment for a given set of statements 
#### The trained model when submitted fetched an accuracy> 90% and a rank 50

The training and data is taken from the website taken from https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/#About

* The first step of the approach needed to solve with the problem is training data visualization 
The data is given in the train.csv file in the repository https://github.com/namantuli18/Sentiment_Predictions/blob/master/train.csv
The data is presented in the comma separated file.
![train.csv](https://github.com/namantuli18/Sentiment_Predictions/blob/master/1.PNG)
A better way to manage the positive and negative sentiment data would be to segregate the separate tweet strings into pos.txt and neg.txt files 
This is done using the separate_pos_neg function in the pos_neg.py file.

* The next step is to create a lexicon file that includes all the tokenized words and create the training and testing data. 

  The tokenizer used is word_tokenize from nltk.tokenize and the lemmatizer used is WordNetLemmatizer from mltk.stem.

  We dump the pickle files into the lexicon.pickle file using the create_features_labels() function.
  
* Now we have the training data with us. The next step would be to create a deep neural network that would ease the task.

  We use basic tensorflow to create a deep neural network with 3 hidden layers and 1500 nodes in each. We train it for 10 epochs and save the sess() to the file model.ckpt to use it further to get results
  
  The lexicon and checkpoint pickle files can be downloaded from the repository.
  
* Next we use the use_neural_network() function in the script.py file to write the results to submission.csv file    
