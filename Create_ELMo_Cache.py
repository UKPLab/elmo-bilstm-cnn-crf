from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings
from util.CoNLL import readCoNLL
import os
import sys
import logging
import time

if len(sys.argv) < 3:
    print("Usage: python Create_ELMo_Cache.py datasetName tokenColumnId [cuda_device]")
    exit()

datasetName = sys.argv[1]
tokenColId = int(sys.argv[2])
cudaDevice = int(sys.argv[3]) if len(sys.argv) >= 4 else -1

elmo_options_file= 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


commentSymbol = None
columns = {tokenColId: 'tokens'}



picklePath = "embeddings/elmo_cache_" + datasetName + ".pkl"

embLookup = ELMoWordEmbeddings(None, elmo_options_file, elmo_weight_file, elmo_cuda_device=cudaDevice)

print("ELMo Cache Generation")
print("Output file:", picklePath)
print("CUDA Device:", cudaDevice)

splitFiles = ['train.txt', 'dev.txt', 'test.txt']

for splitFile in splitFiles:
    inputPath = os.path.join('data', datasetName, splitFile)

    print("Adding file to cache: "+inputPath)
    sentences = readCoNLL(inputPath, columns, commentSymbol)
    tokens = [sentence['tokens'] for sentence in sentences]

    start_time = time.time()
    embLookup.addToCache(tokens)
    end_time = time.time()
    print("%s processed in %.1f seconds" % (splitFile, end_time - start_time))
    print("\n---\n")

print("Store file at:", picklePath)
embLookup.storeCache(picklePath)


