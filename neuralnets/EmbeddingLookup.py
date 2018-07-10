from abc import ABC, abstractmethod
import urllib.request as urllib2
import urllib.parse as urlparse
from urllib.request import urlretrieve
import sys
import os
import logging
import gzip
import numpy as np

class EmbeddingLookup(ABC):
    @abstractmethod
    def sentenceLookup(self, sentence):
        """Maps the tokens in sentence to their according word vectors"""
        pass

    @abstractmethod
    def getIdentifier(self):
        """Returns a unique identifier for this lookup function"""
        pass

    def readEmbeddings(self, embeddingsPath):
        filename = os.path.basename(embeddingsPath)
        if not os.path.isfile(embeddingsPath):
            if filename in ['komninos_english_embeddings.gz', 'levy_english_dependency_embeddings.gz', 'reimers_german_embeddings.gz']:
                self.getEmbeddings(filename, embeddingsPath)
            else:
                print("The embeddings file %s was not found" % embeddingsPath)
                exit()

        # :: Read in word embeddings ::
        logging.info("Read file: %s" % embeddingsPath)
        word2Idx = {}
        embeddings = []
        embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
        embeddingsDimension = None

        for line in embeddingsIn:
            split = line.rstrip().split(" ")
            word = split[0]

            if embeddingsDimension==None:
                embeddingsDimension = len(split) - 1

            if (len(split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
                print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                continue

            if len(word2Idx)==0:  # Add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(embeddingsDimension)
                embeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
                embeddings.append(vector)

            vector = np.array([float(num) for num in split[1:]])

            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)

        return word2Idx, embeddings

    def getEmbeddings(self, filename, savePath):
        if not os.path.isfile(savePath):
            self.download("https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/" + filename, savePath)

    def download(self, url, savePath, silent=False):
        filename = os.path.basename(urlparse.urlparse(url).path) or 'downloaded.file'

        def get_size():
            meta = urllib2.urlopen(url).info()
            meta_func = meta.getheaders if hasattr(
                meta, 'getheaders') else meta.get_all
            meta_length = meta_func('Content-Length')
            try:
                return int(meta_length[0])
            except:
                return 0

        def kb_to_mb(kb):
            return kb / 1024.0 / 1024.0

        def callback(blocks, block_size, total_size):
            current = blocks * block_size
            percent = 100.0 * current / total_size
            line = '[{0}{1}]'.format(
                '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
            status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
            sys.stdout.write(
                status.format(
                    percent, line, kb_to_mb(current), kb_to_mb(total_size)))


        logging.info(
            'Downloading: {0} ({1:3.1f} MB)'.format(url, kb_to_mb(get_size())))
        try:
            (savePath, headers) = urlretrieve(url, savePath, None if silent else callback)
        except:
            os.remove(savePath)
            raise Exception("Can't download {0}".format(savePath))
        else:
            print()
            logging.info('Downloaded to: {0}'.format(savePath))

        return savePath