# BiLSTM-CNN-CRF with ELMo-Representations for Sequence Tagging

This repository is an extension of my [BiLSTM-CNN-CRF implementation](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/).

It integrates the ELMo representations from the publication [Deep contextualized word representations](http://arxiv.org/abs/1802.05365) (Peters et al., 2018) into the [BiLSTM-CNN-CRF architecture](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/) and can improve the performance significantly for different sequence tagging tasks.


The system is **easy to use**, optimized for **high performance**, and highly **configurable**.

**Requirements:**
* Python 3.6 - lower versions of Python do not work
* AllenNLP 0.5.1 - to compute the ELMo representations
* Keras 2.2.0 - For the creation of BiLSTM-CNN-CRF architecture
* Tensorflow 1.8.0 - As backend for Keras (other backends are untested.

**Note:** This implementation might be incompatible with different (e.g. more recent) versions of the frameworks. See [docker/requirements.txt](docker/requirements.txt) for a full list of all Python package requirements.



# Documentation
This code is an extension of the [emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/) implementation. Must examples can be used with only slight adaptation. Also please see that repository for an explanation about the definition of the datasets, the configuration of the hyperparameters, how to use it for multi-task learning, or how to create custom features.

Most aspects from [emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/) work the same in this implementation.

# Citation
This repository contains experimental software and is under active development. I working on the create of additional material to explain the usage of ELMo representations in state-of-the-art classifier.

Until then, if you find the implementation useful, please cite the following paper: [Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging](https://arxiv.org/abs/1707.09861)

```
@InProceedings{Reimers:2017:EMNLP,
  author    = {Reimers, Nils, and Gurevych, Iryna},
  title     = {{Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging}},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {09},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  pages     = {338--348},
  url       = {http://aclweb.org/anthology/D17-1035}
}
```



Contact person: Nils Reimers, reimers@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/ https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.



# Setup
In order to run the code, Python 3.6 or higher is required. The code is based on Keras 2.2.0 and as backend I recommend Tensorflow 1.8.0. I cannot ensure that the code works with different versions for Keras / Tensorflow or with different backends for Keras.

### Installation using virtualenv
To get the ELMo representations, AllenNLP is required. The [AllenNLP installation instructions](https://github.com/allenai/allennlp) describe a nice way how to setup a virtual enviromnent with the correct Python version.

[Conda](https://conda.io/) can be used set up a virtual environment with the version of Python required (3.6).

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```bash
    conda create -n elmobilstm python=3.6
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to this code.

    ```bash
    source activate elmobilstm
    ```

### Installing the dependencies with pip

You can use `pip` to install the dependencies.

```bash
pip install allennlp==0.5.1 tensorflow==1.8.0 Keras==2.2.0
```

In [docker/requirements.txt)](docker/requirements.txt) you find a full list of all used packages. You can install it via:
```bash
pip install -r docker/requirements.txt
```

### Installation using docker

The [docker-folder](docker/) contains an example how to create a Docker image that contains all required dependencies. It can be used to run your code within that container. See the docker-folder for more details.

# Training
See `Train_Chunking.py` for an example how to train and evaluate this implementation. The code assumes a CoNLL formatted dataset like the CoNLL 2000 dataset for chunking.

For training, you specify the datasets you want to train on:
```
datasets = {
    'conll2000_chunking':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'chunk_BIO',                              #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}
```

For more details, see the [emnlp2017-bilstm-cnn-crf implementation](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/).

# Computing ELMo representations
The computation of ELMo representations is computationally expensive. A CNN is used to map the characters of a token to a dense vectors. These dense vectors are then fed through two BiLSTMs. The representation of each token and the two outputs of the BiLSTMs are used to form the final context-dependend word embedding.

In order speed-up the training, we pre-compute the context dependend word embeddings for all sentences in our training, development, and test set. Hence, instead of passing word indizes to the BiLSTM-CRF architecture, we pass the final 1024 dimensional embeddings to the architecture.

The relevant code looks like:
```
embLookup = ELMoWordEmbeddings(embeddings_file, elmo_options_file, elmo_weight_file, elmo_mode, elmo_cuda_device)
pickleFile = perpareDataset(datasets, embLookup)
```

The `ELMoWordEmbeddings` provides methods for the efficient compuation of ELMo representations. It has the following parameters:
* `embeddings_file`: The ELMo paper concatenates traditional word embeddings, like GloVe, with the context dependent embeddings. With `embeddings_file` you can pass a path to a pre-trained word embeddings file. You can set it to `none` if you don't want to use traditional word embeddings.
* `elmo_options_file` and `elmo_weight_file`: AllenNLP provides different pretrained ELMo models.
* `elmo_mode`: Set to `average` if you want all 3 layers to be averaged. Set to `last` if you want to use only the final layer of the ELMo language model.
* `elmo_cuda_device`: Can be set to the ID of the GPU which should compute the ELMo embeddings. Set to `-1` to run ELMo on the CPU. Using a GPU drastically improves the computational time.

The `perpareDataset` method requires the `embLookup`-object as an argument. It then iterates through all sentences in your dataset, computes the ELMo embeddings, and stores it in a pickle-file in the `pkl/` folder.

## Pre-compute ELMo embeddings once
The `ELMoWordEmbeddings` class implements a caching mechansim for a quick lookup of sentences => context dependent word representations for all tokens in the sentence.

You can run `Create_ELMo_Cache.py` to iterate through all you sentences in your dataset and create the ELMo embeddings for those. It stores these embeddings in the file `embeddings/elmo_cache_[DatasetName].pkl`.

Once you create such a cache, you can load those in your experiments:
```
embLookup = ELMoWordEmbeddings(embeddings_file, elmo_options_file, elmo_weight_file, elmo_mode, elmo_cuda_device)
embLookup.loadCache('embeddings/elmo_cache_conll2000_chunking.pkl')
pickleFile = perpareDataset(datasets, embLookup)
```

If a sentence is in the cache, the cached representations for all tokens in that sentence are used. This requires the computation of the ELMo embeddings for a dataset must only be done once.

*Note:* The cache file can become rather large, as 3*1024 float numbers per token must be stored. The cache file requires about 3.7 GB for the CoNLL 2000 dataset on chunking with about 13.000 sentences.


## Issues, Feedback, Future Development
This repository is under active development as I'm currently running several experiments that involve ELMo embeddings.

If you have questions, feedback or find bugs, please send an email to me: reimers@ukp.informatik.tu-darmstadt.de
