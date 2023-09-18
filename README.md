# random neural nets

Implementations of a random selection of artificial neural net based models and methods.

## Python version

Development is done using [`pyenv`](https://realpython.com/intro-to-pyenv/), pinning the python version to the one in the file `.python-version`.

## Installation

On Linux:

    git clone https://github.com/eschmidt42/random-neural-net-models.git
    cd random-neural-net-models
    make install

## Usage

See jupyter notebooks in `nbs/` for:
* perceptron: `perceptron.ipynb`
* backpropagation: `backpropagation_rumelhart1986.ipynb`
* convolution: `convolution_lecun1990.ipynb`
* cnn autoencoder:
    * mnist: `cnn_autoencoder_fastai2022.ipynb`
    * fashion mnist: `cnn_autoencoder_fastai2022_fashion.ipynb`
* variational autoencoder:
    * dense: `dense_variational_autoencoder_fastai2022.ipynb`
    * cnn+dense: `cnn_variational_autoencoder_fastai2022.ipynb`