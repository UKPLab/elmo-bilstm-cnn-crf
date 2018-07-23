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

    def sentenceLookup(self, sentences):
        elmo_vectors = None

        # :: Elmo ::
        if self.elmo_mode is not None:
            elmo_vectors = self.getElmoEmbedding(sentences)



        # :: Word Embedding ::
        tokens_vectors = None
        if self.embeddingsPath is not None:
            if self.word2Idx is None or self.embeddings is None:
                self.word2Idx, self.embeddings = self.readEmbeddings(self.embeddingsPath)

            tokens_vectors = []
            for sentence in sentences:
                per_token_embedding = []
                for token in sentence['tokens']:
                    vecId = self.word2Idx['UNKNOWN_TOKEN']

                    if token in self.word2Idx:
                        vecId = self.word2Idx[token]
                    elif token.lower() in self.word2Idx:
                        vecId = self.word2Idx[token.lower()]
                    per_token_embedding.append(self.embeddings[vecId])

                tokens_vectors.append(per_token_embedding)


        out_vectors = []
        if elmo_vectors is not None and tokens_vectors is not None:
            for idx in range(len(sentences)):
                out_vectors.append(np.append(tokens_vectors[idx], elmo_vectors[idx], axis=1).astype(np.float32))
        elif elmo_vectors is not None:
            out_vectors = elmo_vectors
        elif tokens_vectors is not None:
            out_vectors = tokens_vectors
        else:
            print("No vector retrieved")
            assert(False)

        return out_vectors

    def applyElmoMode(self, elmo_vectors):
        if self.elmo_mode=='average':
            return np.average(elmo_vectors, axis=0).astype(np.float32)
        elif self.elmo_mode=='last':
            return np.asarray[-1, :, :]
        elif isinstance(self.elmo_mode, int):
            return elmo_vectors[int(self.elmo_mode), :, :]
        else:
            print("Unknown ELMo mode")
            assert (False)

    def getElmoEmbedding(self, sentences):
        if len(self.lazyCacheFiles) > 0:
            self._loadLazyCache()

        elmo_embeddings = []
        non_cached_sentences = []
        non_cached_sentences_indices = []

        # :: Lookup cached sentences ::
        for sentence in sentences:
            tokens = sentence['tokens']
            if len(self.cache) > 0 and tuple(tokens) in self.cache:
                elmo_embeddings.append(self.applyElmoMode(self.cache[tuple(tokens)]))
            else:
                non_cached_sentences.append(tokens)
                non_cached_sentences_indices.append(len(elmo_embeddings))
                elmo_embeddings.append(None)

        # :: Compute ELMo on the fly ::
        if len(non_cached_sentences) > 0:
            if self.elmo is None:
                self.loadELMo()

            idx = 0
            for elmo_vectors in self.elmo.embed_sentences(non_cached_sentences):
                assert(elmo_embeddings[non_cached_sentences_indices[idx]] == None)
                elmo_embeddings[non_cached_sentences_indices[idx]] = self.applyElmoMode(elmo_vectors)
                idx += 1

        return elmo_embeddings

    def getIdentifier(self):
        """Returns a unique identifier for this lookup function"""
        return "ELMoWordEmbeddings_"+self.embeddingName+"_"+str(self.elmo_mode)

    def loadELMo(self):
        self.elmo = ElmoEmbedder(self.elmo_options_file, self.elmo_weight_file, self.elmo_cuda_device)

    def loadCache(self, inputPath):
        self.lazyCacheFiles.append(inputPath)

    def storeCache(self, outputPath):
        f = open(outputPath, 'wb')
        pkl.dump(self.cache, f, -1)
        f.close()

    def addToCache(self, sentences):
        if self.elmo is None:
            self.loadELMo()

        idx = 0
        for elmoEmbedding in self.elmo.embed_sentences(sentences):
            sentence = tuple(sentences[idx])
            self.cache[sentence] = elmoEmbedding

            idx += 1

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

