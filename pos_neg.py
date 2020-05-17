import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer
import pickle
lemmatizer=WordNetLemmatizer()

df=pd.read_csv('train.csv')
#print(df.head())
def separate_pos_neg(df):
	for count,row in enumerate(df['label']):
		if row == 0:
			#print(row,df['tweet'][count])
			with open('pos.txt','a',encoding='UTF-8') as f:
				#print(df['tweet'][count])
				f.write("{}\n".format(df['tweet'][count]))

		elif row == 1:
			with open('neg.txt','a',encoding='UTF-8') as f:
				f.write("{}\n".format(df['tweet'][count]))


#separate_pos_neg(df)

def create_lexicon(pos,neg):
	lexicon=[]
	with open(pos,'r',encoding='UTF-8') as f:
		contents=f.readlines()
		for lines in contents:
			all_words=word_tokenize(lines.lower())
			lexicon+=list(all_words)

	with open(neg,'r',encoding='UTF-8') as f:
		contents=f.readlines()
		for lines in contents:
			all_words=word_tokenize(lines.lower())
			lexicon+=list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	#print(lexicon)
	l1=[]
	for individual in lexicon:  
		if individual!='#' and individual[0]!='/':
			l1.append(individual)

	l2=[]
	w_counts=Counter(lexicon)
	for i in w_counts:
		if w_counts[i]>5 and w_counts[i]<1000:
			l2.append(i)
	print(l2)
	return l2
#create_lexicon('pos.txt','neg.txt')

def sample_handling(sample,lexicon,classification):
	featureset=[]
	with open(sample,'r',encoding='UTF-8') as f:
		contents =f.readlines()
		for l in contents:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			#features = list(features)
			#print(features)
			featureset.append([features,classification])
			#print(featureset)
	return featureset

def create_features_labels(pos,neg,test_size=0.1):
	lexicon=create_lexicon(pos,neg)
	features=[]
	features+=sample_handling('pos.txt',lexicon,[1,0])
	features+=sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(features)
	features=np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

if __name__ == "__main__":
	train_x,train_y,test_x,test_y=create_features_labels('pos.txt','neg.txt')
	with open('lexicon.pickle',"wb") as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
