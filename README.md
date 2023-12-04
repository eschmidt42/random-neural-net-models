# random neural nets

Implementations of a random selection of artificial neural net based models and methods.

## Python version

Development is done using [`pyenv`](https://realpython.com/intro-to-pyenv/), pinning the python version to the one in the file `.python-version`.

## Installation (on Linux)

Package + notebooks:

    git clone https://github.com/eschmidt42/random-neural-net-models.git
    cd random-neural-net-models
    make install

Package only:

    pip install random-neural-net-models

## Usage

See jupyter notebooks in [`nbs/`](https://github.com/eschmidt42/random-neural-net-models/tree/main/nbs) for:
* perceptron: `perceptron.ipynb`
* backpropagation: `backpropagation_rumelhart1986.ipynb`
* convolution: `convolution_lecun1990.ipynb`
* cnn autoencoder:
    * mnist: `cnn_autoencoder_fastai2022.ipynb`
    * fashion mnist: `cnn_autoencoder_fastai2022_fashion.ipynb`
* variational autoencoder:
    * dense: `dense_variational_autoencoder_fastai2022.ipynb`
    * cnn+dense: `cnn_variational_autoencoder_fastai2022.ipynb`
* optimizers: `stochastic_optimization_methods.ipynb`
* resnet: `resnet_fastai2022.ipynb`
* unet: `unet_fastai2022.ipynb`
* diffusion (unet + noise): `diffusion_fastai2022.ipynb`
* [mingpt](https://github.com/karpathy/minGPT):
    * `mingpt_sort.ipynb`
    * `mingpt_char.ipynb`
    * `mingpt_adder.ipynb`
