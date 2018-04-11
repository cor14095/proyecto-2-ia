## 
## spamer.py
## 
## By: Alejandro 'Perry' Cortes
## 
## This script categorize messsages base on the corpus train file
## 

## Imports

import random
from collections import Counter

## Functions

def corpusParser(pathToFile):
    # corpus parser function will parse the curpus file
    # and return a list of spam and ham.
    spam = []
    ham = []
    message = []
    msgLine = []
    lineCounter = 0
    
    with open(pathToFile, 'r') as file:
        for line in file:
            lineCounter += 1
            spamOrHam = line.split(None, 1)[0]
            message = line.split(spamOrHam, 1)[1]
            cleanMessage = ""
            for word in message.split():
                if ((word != '') and (len(word) > 3)):
                    # Then this is a usable word.
                    cleanMessage += word + ' '
            # print (cleanMessage)
            if (spamOrHam == "ham"):
                ham.append(cleanMessage)
            else: 
                spam.append(cleanMessage)
    return spam, ham, lineCounter
    
def splitLists(msgList):
    # This function will split the given list
    # returning a train, validate and test list
    trainList = []
    testList = []
    valList = []
    
    for message in msgList:
        msgProbability = random.random()
        if (msgProbability >= 0.8):
            trainList.append(message)
        elif (msgProbability >= 0.9):
            testList.append(message)
        else:
            valList.append(message)
            
    return trainList, testList, valList

def msgClassifier(message, k, wordSize, wordCounter):
    # This function will classifie the message based on 
    # a given k.
    probability = 0
    # print (message)
    for word in message.split():
        # input("word: {} wordCount: {}".format(word, wordCounter[word]))
        probability += (wordCounter[word] + k) / float(sum(wordCounter.values()) + k * wordSize)
    # print(probability)
    return probability

def wordCount(spamTrain, hamTrain, getList = False):
    # word count will return the amount of spam and ham 
    # for the given corpus.
    spamCounter = Counter()
    hamCounter = Counter()
    
    for message in spamTrain:
        for word in message.split():
            spamCounter[word] += 1
    for message in hamTrain:
        for word in message.split():
            hamCounter[word] += 1
    
    if (getList):
        return list(set(spamCounter + hamCounter))
    else:
        return spamCounter, hamCounter
        
def validator(valData, hamTrain, spamTrain, wordSetSize, spamProb, hamProb, dataType):
    errorPerc = 100
    k = 0
    while errorPerc > 10:
        k += 0.1
        valError = 0
        for line in valData:
            result = spamer(k, hamTrain, spamTrain, wordSetSize, line, spamProb, hamProb)
            if (result != dataType):
                valError += 1
        errorPerc = float(valError / float(len(valData))) * 100
        print ("K: {} - {}".format(k, errorPerc))
    
    return k

def spamer(k, hamTrain, spamTrain, wordSize, message, spamProb, hamProb):
    # Spamer will return if a word is spam or not.
    spamWordCount, hamWordCount = wordCount(spamTrain, hamTrain)
    # print ("spam prob: {} \tham prob: {}".format(spamProb, hamProb))
    
    # print("spam:")
    spamProb = msgClassifier(message, k, wordSize, spamWordCount)
    # print("ham")
    hamProb = msgClassifier(message, k, wordSize, hamWordCount)

    # input ("spam prob: {} \tham prob: {}".format(spamProb, hamProb))

    if (hamProb >= spamProb):
        return "ham"
    else:
        return "spam"
    
## Main program

def main():
    # Variables
    corpusLines     = 0
    spamProbability = 0
    hamProbability  = 0
    spam            = []
    spamTrain       = []
    spamVal         = []
    spamTest        = []
    ham             = []
    hamTrain        = []
    hamVal          = []
    hamTest         = []
    filePath        = 'test_corpus.txt'
    wordSet         = list()
    valSet          = list()
    testSet         = list()
    k               = 1
    
    # Get all sets and variables needed
    spam, ham, corpusLines = corpusParser(filePath)
    spamTrain, spamTest, spamVal = splitLists(spam)
    hamTrain, hamTest, hamVal = splitLists(ham)
    wordSet = wordCount(spamTrain, hamTrain, True)
    spamProbability = float( (len(spam) / corpusLines))
    hamProbability = float( (len(ham) / corpusLines))
    # print ("{} {}".format(spamProbability, hamProbability))
    
    # Validate the spamer
    # k = validator(hamVal, hamTrain, spamTrain, len(wordSet),
    #                 spamProbability, hamProbability, "ham")

    # k = validator(spamVal, hamTrain, spamTrain, len(wordSet),
    #                 spamProbability, hamProbability, "spam")

    k = 2.6
    
    # Look for the file to test:

    # classify file
    messages = []
    
    try:
        # Read input file.
        file = open("input_classified.txt", 'w')
        with open("input.txt", 'r') as testFile:
            for line in testFile:
                cleanMessage = ""
                for word in line.split():
                    if ((word != '') and (len(word) > 3)):
                        # Then this is a usable word.
                        cleanMessage += word + ' '
                classification = spamer(k, hamTrain, spamTrain, len(wordSet), cleanMessage, 
                spamProbability, hamProbability)
                message = "{}\t{}\n".format(classification, line)
                file.write(message)
        file.close()
    except:
        print ("No file found.")
        
    return 0

if __name__ == "__main__":
    # execute only if run as a script
    main()
    