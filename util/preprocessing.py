from __future__ import (division, absolute_import, print_function, unicode_literals)
import os
import numpy as np
import os.path
import logging
from .CoNLL import readCoNLL

import sys
import pickle as pkl


def perpareDataset(datasets, embeddingsClass, padOneTokenSentence=True):
    embeddingsName = embeddingsClass.getIdentifier()
    embeddingsFct = embeddingsClass.sentenceLookup
    pklName = "_".join(sorted(datasets.keys()) + [embeddingsName])
    outputPath = 'pkl/' + pklName + '.pkl'

    if os.path.isfile(outputPath):
        logging.info("Using existent pickle file: %s" % outputPath)
        return outputPath

    casing2Idx = getCasingVocab()

    mappings = {'tokens': {}, 'casing': casing2Idx}
    pklObjects = {'mappings': mappings, 'datasets': datasets, 'data': {}}

    for datasetName, dataset in datasets.items():
        datasetColumns = dataset['columns']
        commentSymbol = dataset['commentSymbol']

        trainData = 'data/%s/train.txt' % datasetName 
        devData = 'data/%s/dev.txt' % datasetName 
        testData = 'data/%s/test.txt' % datasetName 
        paths = [trainData, devData, testData]

        logging.info("\n:: Transform "+datasetName+" dataset ::")
        pklObjects['data'][datasetName] = createPklFiles(paths, mappings, datasetColumns, commentSymbol, padOneTokenSentence)

        for datasplit in pklObjects['data'][datasetName]:
            addEmbeddings(pklObjects['data'][datasetName][datasplit], embeddingsFct, padOneTokenSentence)


    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("\n\nDONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath

def addEmbeddings(sentences, embeddingsFct, padOneTokenSentence=True):
    print("\n\n:: Lookup embeddings and tokens ::")
    sentCnt = 0
    total_size = len(sentences)

    for sentence in sentences:
        sentence['word_embeddings'] = embeddingsFct(sentence['tokens'])

        if padOneTokenSentence and len(sentence['word_embeddings'])==1:
            sentence['word_embeddings'] = np.append(sentence['word_embeddings'],
                                                    np.zeros((1, len(sentence['word_embeddings'][0]))), axis=0)

        sentCnt += 1
        percent = 100.0 * sentCnt / total_size
        line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} sentences'
        sys.stdout.write(status.format(percent, line, sentCnt, total_size))


def loadDatasetPickle(embeddingsPickle):
    """ Loads the cPickle file, that contains the word embeddings and the datasets """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()

    return pklObjects['mappings'], pklObjects['data']


def addCharInformation(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)

def addCasingInformation(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))
       
       
def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def getCasingVocab():
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}


def createMatrices(sentences, mappings, padOneTokenSentence):
    data = []
    for sentence in sentences:
        row = {name: [] for name in list(mappings.keys())}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
                    
            for entry in sentence[mapping]:                
                if mapping.lower() == 'tokens':
                    idx = entry
                elif mapping.lower() == 'characters':  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])                           
                                      
                else:
                    idx = str2Idx[entry]
                                    
                row[mapping].append(idx)



        if len(row['tokens'])==1 and padOneTokenSentence:
            for mapping, str2Idx in mappings.items():
                if mapping.lower()=='tokens':
                    pass
                elif mapping.lower()=='characters':
                    row['characters'].append([0])
                else:
                    row[mapping].append(0)

        data.append(row)

    return data
    
  
  
def createPklFiles(datasetFiles, mappings, cols, commentSymbol, padOneTokenSentence):
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol)
   
    extendMappings(mappings, trainSentences+devSentences+testSentences)


    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset
    
    addCharInformation(trainSentences)
    addCasingInformation(trainSentences)
    
    addCharInformation(devSentences)
    addCasingInformation(devSentences)
    
    addCharInformation(testSentences)   
    addCasingInformation(testSentences)

    logging.info(":: Create Train Matrix ::")
    trainMatrix = createMatrices(trainSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Dev Matrix ::")
    devMatrix = createMatrices(devSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Test Matrix ::")
    testMatrix = createMatrices(testSentences, mappings, padOneTokenSentence)

    
    data = {
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def extendMappings(mappings, sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens') #No need to map tokens

    for sentence in sentences:
        for name in sentenceKeys:
            if name not in mappings:
                mappings[name] = {'O':0} #'O' is also used for padding

            for item in sentence[name]:              
                if item not in mappings[name]:
                    mappings[name][item] = len(mappings[name])