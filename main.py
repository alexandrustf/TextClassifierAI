from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import loadDataSet
import numpy
import re
import tfidf
from gensim.models import Word2Vec


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


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
    words = re.sub("[^a-zA-ZăâîșțÂÎȘȚ]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in stop_words]
    return cleaned_text


def generate_bow(allsentences, vocab):
    # print(allsentences)
    datas = dict()
    for index, sentence in enumerate(allsentences):
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
        datas[index] = numpy.array(bag_vector)
        # print("{0} \n{1}\n".format(sentence, numpy.array(bag_vector)))
    return datas


def filter_words(allsentences):
    return tokenize(allsentences)


def word2vec_representation(articles_text): # documented from https://stackabuse.com/implementing-word2vec-with-gensim-library-in-python/
    all_sentences = []
    for article_text in articles_text:
        processed_article = word_extraction(article_text)
        all_sentences.append(processed_article)

    word2vec = Word2Vec(all_sentences, min_count=2)
    return word2vec


trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels = loadDataSet.loadMOROCODataSamples("train")

vocabulary = filter_words(trainSamples)

print('Vocabulary is: ')
print(vocabulary)
print('Length of vocabulary is: ' + str(len(vocabulary)))
print()
bow_dict = generate_bow(trainSamples, vocabulary)
print(bow_dict)
print('TF-IDF vectorial representation is: ')
print(tfidf.tf_idf_representation(trainSamples))
word2vec_model = word2vec_representation(trainSamples)
print(word2vec_model.wv.vocab)
v1 = word2vec_model.wv['spus']
print(v1)
sim_words = word2vec_model.wv.most_similar('abilităților')
print(sim_words)
