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
from gensim.models import Word2Vec
import logging
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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


def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


def word_extraction(sentence):
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
    stop_words.add('–')
    stop_words.add(',')
    stop_words.add('%')
    stop_words.add('$')
    # print(sentence)
    words = re.sub("[^a-zA-ZăâîșțÂÎȘȚ]", " ", str(sentence)).split()
    cleaned_text = [w.lower() for w in words if w not in stop_words]
    return cleaned_text


def word2vec_representation(articles_text): # documented from https://stackabuse.com/implementing-word2vec-with-gensim-library-in-python/
    all_sentences = []
    for article_text in articles_text:
        processed_article = word_extraction(article_text)
        all_sentences.append(processed_article)

    word2vec = Word2Vec(all_sentences, min_count=2)
    return word2vec


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
#here are just a statiscs of input (it is not necessary from our classifier)
cnt_pro = df['Category'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.xticks(rotation=90)
# plt.show()

df['Text'] = df['Text'].apply(cleanText)

print('ceva')

X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Category, random_state=0, test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test
len(all_data)
# word2vec representation
word2vec_model = word2vec_representation(X_train)
word2vec_model.init_sims(True)
wv = word2vec_model.wv
print("\n Training the word2vec model...\n")

train, test = train_test_split(df, test_size=0.3, random_state=42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values

X_train_word_average = word_averaging_list(wv, train_tokenized)
X_test_word_average = word_averaging_list(wv, test_tokenized)

logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=2000)
logreg = logreg.fit(X_train_word_average, train['Category'])
y_pred = logreg.predict(X_test_word_average)
print('accuracy %s' % accuracy_score(y_pred, test.Category))