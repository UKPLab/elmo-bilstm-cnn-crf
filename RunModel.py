#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
import nltk
import sys
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation, addEmbeddings
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]


# :: Load the model ::
lstmModel = ELMoBiLSTM.loadModel(modelPath)


# :: Get the embedding lookup class ::
embLookup = lstmModel.embeddingsLookup
embLookup.elmo_cuda_device = -1         #Cuda device for pytorch - elmo embedding, -1 for CPU

# :: Read input ::
with open(inputPath, 'r') as f:
    text = f.read()


# :: Prepare the input ::
sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
addCharInformation(sentences)
addCasingInformation(sentences)

# :: Map casing and character information to integer indices ::
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Perform the word embedding / ELMo embedding lookup ::
addEmbeddings(dataMatrix, embLookup.sentenceLookup)


# :: Use the model to tag the input ::
tags = lstmModel.tagSentences(dataMatrix)

# :: Output to stdout ::
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

        print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    print("")