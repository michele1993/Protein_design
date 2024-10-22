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
## Generative protein sequence base model 
I experimented with [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) base model for protein sequences, which is available on Hugging Face. To do that, I install [Hugging Face package](https://huggingface.co/docs/transformers/installation) for Pytorch (CPU-only version).
``` sh
pip install transformers
```
For CPU-only uses,
```sh
pip install 'transformers[torch]'
```
The model also provides a fine-turning option, which allows you to fine-tune the model to a specific dataset. To do that, I dowloaded the `run_clm.py` file from the specified Hugging face [repository](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py).
```sh
wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/language-modeling/run_clm.py
```

**Note**: When trying to run `python run_clm.py` I encoutered an error with my Hugging face version. This was solved by installing Hugging face from source.
```sh
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

