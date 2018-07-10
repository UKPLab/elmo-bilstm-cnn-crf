#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
import nltk
import sys
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation, addEmbeddings, readCoNLL
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings


if len(sys.argv) < 3:
    print("Usage: python RunModel_CoNLL_Format.py modelPath inputPathToConllFile")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]
inputColumns = {0: "tokens"}

embeddings_file = 'embeddings/komninos_english_embeddings.gz'
elmo_options_file= 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
elmo_mode = 'average'

# :: Load the model ::
lstmModel = ELMoBiLSTM.loadModel(modelPath)

# :: Prepare the input ::
sentences = readCoNLL(inputPath, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)

# :: Map casing and character information to integer indices ::
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Perform the word embedding / ELMo embedding lookup ::
embLookup = ELMoWordEmbeddings(embeddings_file, elmo_options_file, elmo_weight_file, elmo_mode)
addEmbeddings(dataMatrix, embLookup.sentenceLookup)


# :: Tag the input ::
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