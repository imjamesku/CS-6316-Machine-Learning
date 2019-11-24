#!/usr/bin/python

import sys
import os
import numpy as np
import collections
import string
from sklearn.naive_bayes import MultinomialNB

###############################################################################


def loadDictionary(path):
    vocab = open(path).read().split()
#     print(vocab)
    N = len(vocab)
    wordToIdxMapping = collections.defaultdict(lambda: N-1)
    for i, word in enumerate(vocab):
        wordToIdxMapping[word] = i
    return wordToIdxMapping


def transfer(fileDj, vocabulary):
    tokens = open(fileDj).read().strip()\
        .replace('loved', 'love').replace('loves', 'love').replace('loving', 'love')\
        .translate(str.maketrans('', '', string.punctuation)).split()
#     print(tokens)
#     print(len(vocabulary))
    BOWDj = [0] * len(vocabulary)
    for word in tokens:
        if word in vocabulary:
            BOWDj[vocabulary[word]] += 1
        else:
            BOWDj[-1] += 1
    return BOWDj


def loadData(Path):
    #     print(os.getcwd())
    wordToIdxMapping = loadDictionary("dictionary.txt")
    Xtrain, ytrain = [], []
    # train
    trainPosPath = Path + '/training_set/pos'
    for file in os.listdir(trainPosPath):
        if file.endswith(".txt"):
            Xtrain.append(transfer(os.path.join(
                trainPosPath, file), wordToIdxMapping))
            ytrain.append(1)
    trainNegPath = Path + '/training_set/neg'
    for file in os.listdir(trainNegPath):
        if file.endswith(".txt"):
            Xtrain.append(transfer(os.path.join(
                trainNegPath, file), wordToIdxMapping))
            ytrain.append(0)
    Xtrain = np.array(Xtrain).T
    ytrain = np.array(ytrain)
#     print(Xtrain.shape)
#     print(Xtrain)
#     print(ytrain)
    Xtest, ytest = [], []
    testPosPath = Path + '/test_set/pos'
    for file in os.listdir(testPosPath):
        if file.endswith(".txt"):
            Xtest.append(transfer(os.path.join(
                testPosPath, file), wordToIdxMapping))
            ytest.append(1)
    testNegPath = Path + '/test_set/neg'
    for file in os.listdir(testNegPath):
        if file.endswith(".txt"):
            Xtest.append(transfer(os.path.join(
                testNegPath, file), wordToIdxMapping))
            ytest.append(0)
    Xtest = np.array(Xtest).T
    ytest = np.array(ytest)
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain, alpha=1.0):
    """
    Xtrain: v*d, where v is the vocab size and d is the number of documents
    """
    vocabSize = Xtrain.shape[0]
    # pos
    Xpos = Xtrain[:, ytrain == 1]
    ypos = ytrain[ytrain == 1]
    lenToalPosDocs = np.sum(Xpos)
    thetaPos = np.array([(np.sum(
        Xpos[i]) + alpha)/(lenToalPosDocs + alpha*vocabSize) for i in range(vocabSize)])
    # neg
    Xneg = Xtrain[:, ytrain == 0]
    yneg = ytrain[ytrain == 0]
    lenTotalNegDocs = np.sum(Xneg)
    thetaNeg = np.array([(np.sum(
        Xneg[i]) + alpha)/(lenTotalNegDocs + alpha*vocabSize) for i in range(vocabSize)])
    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    logThetaPos = np.log(thetaPos)
    logThetaNeg = np.log(thetaNeg)
    numOfDocs = Xtest.shape[1]
    vocabSize = Xtest.shape[0]
    for i in range(numOfDocs):
        freqs = Xtest[:, i]
        posLogP, negLogP = 0, 0
        for j in range(vocabSize):
            posLogP += freqs[j] * logThetaPos[j]
            negLogP += freqs[j] * logThetaNeg[j]
        # print('pos={}, neg={}'.format(posLogP, negLogP))
        if posLogP >= negLogP:
            yPredict.append(1)
        else:
            yPredict.append(0)
    # print(yPredict)
    yPredict = np.array(yPredict)
    Accuracy = yPredict[yPredict == ytest].shape[0]/numOfDocs
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain.T, ytrain)
    score_sklearn = clf.score(Xtest.T, ytest)
    ypredict_sklearn = clf.predict(Xtest.T)
    return Accuracy


def naiveBayesBernFeature_train(Xtrain, ytrain):
    """
    Xtrain: v*d, where v is the vocab size and d is the number of documents
    """
    vocabSize = Xtrain.shape[0]
    NumDocs = Xtrain.shape[1]
    # pos
    Xpos = Xtrain[:, ytrain == 1]
    thetaPosTrue = np.zeros(vocabSize)
#     print('pos')
    for i in range(vocabSize):
        wordVector = Xpos[i]
#         print('wordVecotr: {}'.format(wordVector))
        thetaPosTrue[i] = (
            wordVector[wordVector > 0].shape[0] + 1) / (Xpos.shape[1] + 2)
    # neg
    Xneg = Xtrain[:, ytrain == 0]
    thetaNegTrue = np.zeros(vocabSize)
#     print('neg')
    for i in range(vocabSize):
        wordVector = Xneg[i]
#         print('wordVecotr: {}'.format(wordVector))
        thetaNegTrue[i] = (
            wordVector[wordVector > 0].shape[0] + 1) / (Xneg.shape[1] + 2)
    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    numDocs = Xtest.shape[1]
    vocabSize = Xtest.shape[0]
    for i in range(numDocs):
        freqs = Xtest[:, i]
        posP = negP = 1
        for j in range(freqs.shape[0]):
            f = freqs[j]
            if f > 0:
                posP *= thetaPosTrue[j]
                negP *= thetaNegTrue[j]
            else:
                posP *= (1-thetaPosTrue[j])
                negP *= (1-thetaNegTrue[j])
        if posP > negP:
            yPredict.append(1)
        else:
            yPredict.append(0)
    yPredict = np.array(yPredict)
    Accuracy = yPredict[yPredict == ytest].shape[0]/numDocs
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(
        Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(
        Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
