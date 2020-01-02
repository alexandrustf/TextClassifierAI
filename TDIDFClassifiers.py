import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import doc2vec
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import loadDataSet

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def tokenize_text(text):
    tokens = []
    stop_words = set(stopwords.words('romanian'))
    stop_words.add('$ne$')
    stop_words.add('.')
    stop_words.add('-')
    stop_words.add('și')
    stop_words.add('„$ne$')
    stop_words.add(')')
    stop_words.add(':')
    stop_words.add('(')
    stop_words.add('(')
    stop_words.add('"$ne$')
    stop_words.add('$ne$')
    stop_words.add('–')
    stop_words.add(',')
    stop_words.add('%')
    stop_words.add('$')
    stop_words.add('“$ne$')
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2 or (word in stop_words):
                continue
            tokens.append(word.lower())
    return tokens


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


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


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the complaint narrative.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled


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


trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels = loadDataSet.loadMOROCODataSamples("train")
validationIDs, validationSamples, validationDialectLabels, validationCategoryLabels = loadDataSet.loadMOROCODataSamples("validation")
testIDs, testSamples, testDialectLabels, testCategoryLabels = loadDataSet.loadMOROCODataSamples("test")

# category: 1.culture 2.finance 3.politics 4.science 5.sports 6.tech dialects: 1.moldavian 2. romanian
trainIDs.extend(testIDs)
trainSamples.extend(testSamples)
trainDialectLabels.extend(testDialectLabels)
trainCategoryLabels.extend(testCategoryLabels)

#gather all datas
df = pd.DataFrame(list(zip(trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels)), columns =['Id', 'Text', 'Dialect', 'Category'])
print(df.shape)

df.index = range(27643)
numberOfWords = df['Text'].apply(lambda x: len(x.split(' '))).sum()

print(numberOfWords)

cnt_pro = df['Category'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.xticks(rotation=90)
# plt.show()

df['Text'] = df['Text'].apply(cleanText)

#DBOW
# X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Category, random_state=0, test_size=0.3)
# X_train = label_sentences(X_train, 'Train')
# X_test = label_sentences(X_test, 'Test')
# all_data = X_train + X_test

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


my_tags = ['culture','finance', 'politics','science', 'sports','tech']
X = df.Text
y = df.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

## Tdf-Idf (a)	Naïve Bayes

# nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#               ])
# nb.fit(X_train, y_train)
#
# from sklearn.metrics import classification_report
# y_pred = nb.predict(X_test)
#
# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred,target_names=my_tags))

## Tdf-Idf Linear Support Vector Machine
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# sgd = Pipeline([('vect', CountVectorizer()),
#                 ('tfidf', TfidfTransformer()),
#                 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
#                ])
# sgd.fit(X_train, y_train)
# y_pred = sgd.predict(X_test)
#
# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred,target_names=my_tags))

## Tdf-Idf Logistic Regression
from sklearn.linear_model import LogisticRegression

# logreg = Pipeline([('vect', CountVectorizer()),
# #                 ('tfidf', TfidfTransformer()),
# #                 ('clf', LogisticRegression(n_jobs=1, C=1e5)),
# #                ])
# # logreg.fit(X_train, y_train)
# #
# # y_pred = logreg.predict(X_test)
# #
# # print('accuracy %s' % accuracy_score(y_pred, y_test))
# # print(classification_report(y_test, y_pred,target_names=my_tags))

