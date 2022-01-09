import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix
import re


# Loading the data set
tweet_data = pd.read_csv("C://Users//user//Downloads//naive bayes//Disaster_tweets_NB.csv",encoding = "ISO-8859-1")

# Data cleansing 
  
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))



tweet_data.text = tweet_data.text.apply(cleaning_text)

# removing empty rows
tweet_data = tweet_data.loc[tweet_data.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

tweet_train, tweet_test = train_test_split(tweet_data, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweet_data.text)

# Defining BOW for all messages
all_tweet_matrix = tweet_bow.transform(tweet_data.text)

# For training messages
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing messages
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
# Evaluation on Test Data
accuracy_score(test_pred_m, tweet_test.target) 
pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

# Evaluation on Train Data
accuracy_score(train_pred_m, tweet_train.target) 
pd.crosstab(train_pred_m, tweet_train.target)

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 35)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
# Evaluation on Test Data after applying laplace
accuracy_score(test_pred_lap, tweet_test.target) 
print(metrics.classification_report(test_pred_lap, tweet_test.target))
pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap
# Evaluation on Train Data after applying laplace
accuracy_score(train_pred_lap, tweet_train.target) 
print(metrics.classification_report(train_pred_lap, tweet_train.target))
pd.crosstab(train_pred_lap, tweet_train.target)
