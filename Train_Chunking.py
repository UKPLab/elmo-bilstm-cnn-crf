from __future__ import print_function
import os
import logging
import sys
import torch
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
sess = tf.Session(config=config)
K.set_session(sess)


##################################################

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'conll2000_chunking':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'chunk_BIO',                              #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}


# :: Transform datasets to a pickle file ::
pickleFile = perpareDataset(datasets)

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
embeddings_file = 'embeddings/komninos_english_embeddings.gz'
elmo_options_file= 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
elmo_mode = 'weighted_average'

#Which GPU to use for ELMo. -1 for CPU
if torch.cuda.is_available():
    elmo_cuda_device = 0
else:
    elmo_cuda_device = -1

embLookup = ELMoWordEmbeddings(embeddings_file, elmo_options_file, elmo_weight_file, elmo_mode, elmo_cuda_device)

# You can use a cache that stores the computed ELMo embeddings.
# This increases the training speed, as ELMo embeddings need to computed only once.
# However, it leads to a significant memory overhead of multiple GB (requires about 24KB per token).
#embLookup.cache_computed_elmo_embeddings = True

# We can add a pre-computed ELMo cache to the class. See Create_ELMo_Cache.py how to pre-compute such a cache.
#embLookup.loadCache('embeddings/elmo_cache_conll2000_chunking.pkl')





######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.5, 0.5)}

model = ELMoBiLSTM(embLookup, params)
model.setMappings(mappings)
model.setDataset(datasets, data)
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)



