# Protein design
This repository aims to analyze, sanitize and use a dataset of protein sequences with validated activities to design new alpha-amylase variants with improved activity. This is done through 4 main steps:
1. Data sanitation, e.g, investigate/remove NaN and duplicate sequences, check each sequence only contains natural amino acids, investigate/remove sequences with out of distribution lenghts.
2. Fine-tune a pretrained model: The repositiory uses the pretraind [protGPT2](https://huggingface.co/nferruz/ProtGPT2) base model and fine-tunes it to the enitre (cleaned) alpha-amylase dataset (i.e., independtly of activities) with SFT.
3. Model alignment: The repositiory uses [DPO](https://huggingface.co/docs/trl/main/dpo_trainer) to align the model towards protein sequences with high activities.
4. Generation: Generate seversal sequenses given the aligned model and pick the 'best' one.      

# Installation
## Virtual Environment

To keep things versioned and segregated from the rest of the system, I use a virtual environment. I use a `conda` virtual environment called
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
Next, I begin by installing PyTorch with the latest CUDA release together with a compatible python release.

``` sh
conda install python=3.12.7
pip3 install torch torchvision torchaudio
```
Next, I install pandas to efficiently read the dataset, which is stored in a `.cvs` file (Note: the data file is not provided in the repository).

``` sh
pip3 install pandas
```
### Generative protein sequence base model 
In order to use [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) base model, I install [Hugging Face package](https://huggingface.co/docs/transformers/installation). However, I had to downgrade to earlier verison of it   
``` sh
pip install transformers==4.45.2
```
due to a potential bug between the lastest realise and the `DPOTrainer` of the `trl` package.

To supervise fine tune protGPT2 to my dataset, I dowloaded the `run_clm.py` file from the specified Hugging face [repository](https://github.com/huggingface/transformers/blob/main).
```sh
wget https://github.com/huggingface/transformers/blob/26a9443dae41737e665910fbb617173e17a0cd18/examples/pytorch/language-modeling/run_clm.py
```
**Note**: the `run_clm.py` file must be downloaded from the past commit relating to the `transformers==4.45.2` release, otherwise it won't work with the version of `transformers` installed in the previous step.

Finally, I installed the latest version of the [TRL](https://huggingface.co/docs/trl/index) to align protGPT2 with DPO.
```sh
pip install trl
```
Additional requirements can be found in the `requirments.txt` file.


