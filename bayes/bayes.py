from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocablist(dataSet): 
    vocabSet = set([])

    for document in dataSet :
        vocabSet = vocabSet | set(document)

    return list(vocabSet)

def setOfWords2Vec(vocablist, inputSet) :
    returnVec = [0] * len(vocablist)

    for word in inputSet :
        if word in vocablist :
            returnVec[vocablist.index(word)] = 1
        else : 
            print "the word %s is not in my vocabulary" % word

    return returnVec