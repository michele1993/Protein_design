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

We will begin with PyTorch. Since I am on mac I just install the default version without CUDA.

``` sh
conda install python=3.9
pip3 install torch torchvision torchaudio
```
