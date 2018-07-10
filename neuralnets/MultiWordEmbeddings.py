from .EmbeddingLookup import EmbeddingLookup
import os
import logging
import gzip
import numpy as np


class MultiWordEmbeddings(EmbeddingLookup):
    def __init__(self, embeddingsPaths):
        self.embeddingsPaths = embeddingsPaths
        self.embeddingNames = [os.path.splitext(os.path.basename(embeddingsPath))[0] for embeddingsPath in embeddingsPaths]
        self.embeddingData = {}


        super().__init__()

    def sentenceLookup(self, sentence):
        """Maps the tokens in sentence to their according word vectors"""
        if len(self.embeddingData) == 0:
            self.initLookups()

        sentence_vectors = []
        for embeddingPath in self.embeddingsPaths:
            word2Idx, embeddings = self.embeddingData[embeddingPath]
            sentence_vector = []
            for token in sentence:
                vecId = word2Idx['UNKNOWN_TOKEN']

                if token in word2Idx:
                    vecId = word2Idx[token]
                elif token.lower() in word2Idx:
                    vecId = word2Idx[token.lower()]
                sentence_vector.append(embeddings[vecId])

            sentence_vectors.append(np.asarray(sentence_vector) )

        #Concatenate all the sentence vectors token wise
        single_sentence_vector = sentence_vectors[0]
        for vectorIdx in range(1, len(sentence_vectors)):
            single_sentence_vector = np.append(single_sentence_vector, sentence_vectors[vectorIdx], axis=1)

        return single_sentence_vector

    def getIdentifier(self):
        """Returns a unique identifier for this lookup function"""
        return "MultiWordEmbeddings_"+"_".join(sorted(self.embeddingNames))

    def initLookups(self):
        for embeddingPath in self.embeddingsPaths:
            self.embeddingData[embeddingPath] = self.readEmbeddings(embeddingPath)



