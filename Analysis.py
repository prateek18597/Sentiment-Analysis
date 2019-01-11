import sys
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

training_data=[]
testing_data=[]

for line in open('./full_train.txt'):
	training_data.append(line);

for line in open('./full_test.txt'):
	testing_data.append(line);

tokenizer=nltk.tokenize.TreebankWordTokenizer();

normalized_train=[]

for data in training_data:
	normalized_train.append(tokenizer.tokenize(data));

normalized_test=[]

for data in testing_data:
	normalized_test.append(tokenizer.tokenize(data));


lemmatizer=nltk.stem.WordNetLemmatizer()

lemmatized_train=[]
lemmatized_test=[]

for data in normalized_train:
	lemmatized_train.append(" ".join(lemmatizer.lemmatize(token) for token in data));

for data in normalized_test:
	lemmatized_test.append(" ".join(lemmatizer.lemmatize(token) for token in data));
 
 # min_df=2,max_df=0.5,ngram_range(1,2)		
tfidf=TfidfVectorizer()

tfidf.fit(lemmatized_train)#building vocabulary

X=tfidf.transform(lemmatized_train)
X_test=tfidf.transform(lemmatized_test)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

lr=LogisticRegression(C=0.75)
lr.fit(X,target)
print ("Training Accuracy for C=%s: %s" 
           % (0.75, accuracy_score(target, lr.predict(X))))
print ("Testing Accuracy for C=%s: %s" 
           % (0.75, accuracy_score(target, lr.predict(X_test))))

statements=["It's not good to see you.","Happy to see you"];
print(lr.predict(statements));
