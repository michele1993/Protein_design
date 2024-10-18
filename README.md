# Protein design
This repository aims to analyze, sanitize and use a dataset of protein sequences with validated activities to design new alpha-amylase variants with improved activity. This follows from a technical exercise I was given for a job interview.

## Virtual Environment

To keep things versioned and segregated from the rest of the system, we should
use a virtual environment. We will use a `conda` virtual environment called
`protein_design` for this project.

``` sh
conda create [-p /optional/prefix] -n protein_design
```

## Installing python packages
First I need to activate the newly activated environment so that the pagakes get installed there,
``` sh
conda activate protein_design
```
To avoid having to manually activate the environment every time I use use
[direnv](https://direnv.net/) (highly recommended!).
Next, I begin by installing PyTorch. Since I am on mac I just install the default version without CUDA .

``` sh
conda install python=3.9
pip3 install torch torchvision torchaudio
```
Next, I install pandas to efficiently read the dataset, which is stored in a `.cvs` file.

``` sh
pip3 install pandas
```

I also experiment with some base model for protein sequences, which are available on Hugging Face. To do that, I also need to install [Hugging Face package](https://huggingface.co/docs/transformers/installation) for Pytorch (CPU-only version).
``` sh
pip install 'transformers[torch]'
```
