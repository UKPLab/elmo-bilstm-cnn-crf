from .EmbeddingLookup import EmbeddingLookup
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder, DEFAULT_OPTIONS_FILE, DEFAULT_WEIGHT_FILE
import pickle as pkl
import os



class ELMoWordEmbeddings(EmbeddingLookup):
    def __init__(self, embeddingsPath, elmo_options_file=DEFAULT_OPTIONS_FILE, elmo_weight_file=DEFAULT_WEIGHT_FILE, elmo_mode='average', elmo_cuda_device=-1):
        self.embeddingsPath = embeddingsPath
        self.embeddingName = os.path.splitext(os.path.basename(embeddingsPath))[0] if embeddingsPath is not None else 'None'
        self.word2Idx = None
        self.embeddings = None

        self.elmo_options_file = elmo_options_file
        self.elmo_weight_file = elmo_weight_file
        self.elmo_cuda_device=elmo_cuda_device

        self.elmo_mode = elmo_mode
        self.elmo = None
        self.cache = {}
        self.lazyCacheFiles = []

        super().__init__()

    def sentenceLookup(self, sentence):
        elmo_vector = None
        if self.elmo_mode is not None:
            elmo_vector = self.getElmoEmbedding(sentence)

            if self.elmo_mode =='average':
                elmo_vector = np.average(elmo_vector, axis=0).astype(np.float32)
            elif self.elmo_mode =='last':
                elmo_vector = np.asarray[-1,:,:]
            elif isinstance(self.elmo_mode, int):
                elmo_vector = elmo_vector[int(self.elmo_mode),:,:]
            else:
                print("Unknown ELMo mode")
                assert(False)

        # :: Word Embedding ::
        tokens_vector = None
        if self.embeddingsPath is not None:
            if self.word2Idx is None or self.embeddings is None:
                self.word2Idx, self.embeddings = self.readEmbeddings(self.embeddingsPath)

            tokens_vector = []
            for token in sentence:
                vecId = self.word2Idx['UNKNOWN_TOKEN']

                if token in self.word2Idx:
                    vecId = self.word2Idx[token]
                elif token.lower() in self.word2Idx:
                    vecId = self.word2Idx[token.lower()]
                tokens_vector.append(self.embeddings[vecId])
            tokens_vector = np.asarray(tokens_vector, dtype=np.float32)

        if elmo_vector is not None and tokens_vector is not None:
            out_vector = np.append(tokens_vector, elmo_vector, axis=1).astype(np.float32)
        elif elmo_vector is not None:
            out_vector = elmo_vector
        elif tokens_vector is not None:
            out_vector = tokens_vector
        else:
            print("No vector retrieved")
            assert(False)

        return out_vector

    def getElmoEmbedding(self, sentence):
        if len(self.lazyCacheFiles) > 0:
            self._loadLazyCache()

        if len(self.cache) > 0 and tuple(sentence) in self.cache:
            return self.cache[tuple(sentence)]

        # :: ELMo embeddings ::
        if self.elmo is None:
            self.elmo = ElmoEmbedder(self.elmo_options_file, self.elmo_weight_file, self.elmo_cuda_device)

        return self.elmo.embed_sentence(sentence)

    def getIdentifier(self):
        """Returns a unique identifier for this lookup function"""
        return "ELMoWordEmbeddings_"+self.embeddingName+"_"+str(self.elmo_mode)

    def loadCache(self, inputPath):
        self.lazyCacheFiles.append(inputPath)

    def storeCache(self, outputPath):
        f = open(outputPath, 'wb')
        pkl.dump(self.cache, f, -1)
        f.close()

    def addToCache(self, sentence):
        self.cache[tuple(sentence)] = self.getElmoEmbedding(sentence)

    def _loadLazyCache(self):
        while len(self.lazyCacheFiles) > 0:
            inputPath = self.lazyCacheFiles.pop()

            if not os.path.isfile(inputPath):
                print("ELMo cache file not found:", inputPath)
                continue

            f = open(inputPath, 'rb')
            loaded_cache = pkl.load(f)
            f.close()

            if len(self.cache) == 0:
                self.cache = loaded_cache
            else:
                self.cache.update(loaded_cache)

