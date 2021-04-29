# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:52:35 2021

@author: acer
"""

###############################################################################
###############################################################################
# importing required libraries
import pandas as pd
#import numpy as np
import os

import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn import metrics

###############################################################################
###############################################################################
#
def get_data():
    path="C:/Users/acer/OneDrive/Documents/GitHub/web-framework/01-fastapi"
    os.chdir(path)
    #print(os.getcwd())  # Prints the current working directory

    #
    DATA_DIR = "data" # indicate magical constansts (maybe rather put it on the top of the script)
    data = pd.read_csv(os.path.join(DATA_DIR, "spam.csv"),delimiter=',',encoding='latin-1')

    #drop irrelevant columns
    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
    return data

'''
def lemma(text):
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in range(0, len(text)):
        review = re.sub('[^a-zA-Z]', ' ', text[i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus    
'''

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    #corpus = []

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    #corpus.append(review)
    return review    


## Applying Countvectorizer
# Creating the Bag of Words model

data = get_data()
data['v2'] = data['v2'].apply(clean_text)

#lemmatize_data = lemma(data['v2'])

'''
def vectorize(lemma_data):
    cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
    X = cv.fit_transform(lemma_data).toarray()
    return X
'''
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(data['v2']).toarray()
#X = cv.fit_transform(lemmatize_data).toarray()
    

def get_label(binary_data):
    y = pd.get_dummies(binary_data)
    y = y.drop('ham',axis = 1)
    return y

#X = vectorize(lemmatize_data) 
y = get_label(data['v1'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

###############################################################################
#Naive Bayes classifier for multinomial models
from sklearn.naive_bayes import MultinomialNB 
nb = MultinomialNB()

nb.fit(X_train, y_train)
y_pred  = nb.predict(X_test)

print('accuracy %s' % metrics.accuracy_score(y_pred, y_test))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
#plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

###############################################################################
#Linear support vector machine

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

sgd.fit(X_train, y_train)
y_pred  = sgd.predict(X_test)

print('accuracy %s' % metrics.accuracy_score(y_pred, y_test))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

###############################################################################
#Logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5)

logreg.fit(X_train, y_train)
y_pred  = logreg.predict(X_test)

print('accuracy %s' % metrics.accuracy_score(y_pred, y_test))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

###############################################################################
#Word2vec embedding and Logistic Regression

import logging
import numpy as np

import gensim
import gensim.downloader as api



#wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
#wv.init_sims(replace=True)

wv = api.load('word2vec-google-news-300')
wv.init_sims(replace=True)

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.key_to_index:
            mean.append(wv.vectors[wv.key_to_index[word]])
            all_words.add(wv.key_to_index[word])

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens

train, test = train_test_split(data, test_size=0.3, random_state = 42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['v2']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['v2']), axis=1).values


X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5,solver='lbfgs', max_iter=500)
logreg = logreg.fit(X_train_word_average, train['v1'])
y_pred = logreg.predict(X_test_word_average)



print('accuracy %s' % metrics.accuracy_score(y_pred, test.v1))
print(metrics.classification_report(test.v1, y_pred))
print(metrics.confusion_matrix(test.v1, y_pred))

###############################################################################
#Doc2vec and Logistic Regression

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
#import gensim
from gensim.models.doc2vec import TaggedDocument
#import re


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled

X_train, X_test, y_train, y_test = train_test_split(data.v2, data.v1, random_state=0, test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test


#all_data[:2]

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_data)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors

train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

logreg = LogisticRegression(n_jobs=1, C=1e5,solver='lbfgs', max_iter=500)
logreg.fit(train_vectors_dbow, y_train)

logreg = logreg.fit(train_vectors_dbow, y_train)
y_pred = logreg.predict(test_vectors_dbow)

print('accuracy %s' % metrics.accuracy_score(y_pred, test.v1))
print(metrics.classification_report(test.v1, y_pred))
print(metrics.confusion_matrix(test.v1, y_pred))

###############################################################################
#BOW with keras
import itertools
import os

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


train_size = int(len(data) * .7)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

train_posts = data['v2'][:train_size]
train_tags = data['v1'][:train_size]

test_posts = data['v2'][train_size:]
test_tags = data['v1'][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)


tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

batch_size = 32
epochs = 2

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

###############################################################################
# Save Our Vectorizer
#import sklearn.external.joblib as extjoblib
import joblib

count_vectorizer = open("count_vectorizer.pkl","wb")
joblib.dump(cv,count_vectorizer)

#Saving Our Model¶
NaiveBayesModel = open("naivebayesmodel.pkl","wb")
joblib.dump(nb,NaiveBayesModel)
NaiveBayesModel.close()

###############################################################################
#load the vectorizer and the model to predict
pkl_filename = "naivebayesmodel.pkl"
model = joblib.load(pkl_filename)

pkl_filename = "count_vectorizer.pkl"
vectorizer = joblib.load(pkl_filename)

sms = pd.DataFrame(["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C\'s"],columns=["v2"])
Y=sms['v2'].apply(clean_text)
Y=vectorizer.transform(Y).toarray()
prediction = model.predict(Y)
prediction[0] 


#yprob = model.predict_proba(Y)
#yprob[-2:].round(2)
###############################################################################
###############################################################################


