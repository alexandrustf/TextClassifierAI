import loadDataSet
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels = loadDataSet.loadMOROCODataSamples("train")
# print(trainSamples)


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

#
# tfA = computeTF(numOfWordsA, bagOfWordsA)
# tfB = computeTF(numOfWordsB, bagOfWordsB)
# idfs = computeIDF([numOfWordsA, numOfWordsB])
# tfidfA = computeTFIDF(tfA, idfs)
# tfidfB = computeTFIDF(tfB, idfs)
# df = pd.DataFrame([tfidfA, tfidfB])


def tf_idf_representation(documents): #maybe we can remove romanian useless words
    print('The number of texts is: ' + str(len(documents)))
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return denselist


if __name__ == '__main__':
    documentA = 'the man went out for a walk'
    documentB = 'the children sat around the fire'
    documents = [documentA, documentB]
    bagOfWordsA = documentA.split(' ')
    bagOfWordsB = documentB.split(' ')
    uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
    print(uniqueWords)

    numOfWordsA = dict.fromkeys(uniqueWords, 0)
    for word in bagOfWordsA:
        numOfWordsA[word] += 1
    numOfWordsB = dict.fromkeys(uniqueWords, 0)
    for word in bagOfWordsB:
        numOfWordsB[word] += 1

    # print(pd.DataFrame(tf_idf_representation(documentA,documentB)))
    print(pd.DataFrame(tf_idf_representation(documents)))
