## Installation for developers
```
# Clone repo
git clone https://github.com/mbazzani/AID-model-training-pipeline

# Create a virtual environment
python -m venv .aid-training

# Activate the virtual environment
.aid-training/Scripts/activate.ps1   # for Windows PowerShell
.aid-training/Scripts/activate.bat   # for Windows Command Prompt
source .aid-training/bin/activate    # for bash

# Install in developer mode
python -m pip install -e AID-model-training-pipeline
```
## Description
This repo is meant to facilitate model training for the Acoustic Species ID project, and allow for rapid, iterative testing of different data and models.

## Usage:
To train the current model (efficientnet4), simply run the model_training.py file. 

## Dataset Conventions
Datasets are stored in the `datasets` folder. If `[dataset]` represents the name of the dataset, then paths to training data are in the following format: `datasets/[dataset]/[dataset]_[descriptor]/`. For example, one possible path is `datasets/cosmos/cosmos_chunked/`. Within every 
such folder, there will be several audio clips of training data, as well as a `labels.csv` file containing the corresponding labels for each file.

## Changing Datasets
To change the dataset, simply change the `DATASET` variable in the `config.py` file. If the path to the dataset does not already exist, you will have to create it, and place the data/labels in `datasets/[dataset]/[dataset]_original/`. Then, run `create_chunked_data()` from `training_data_processing.py`

## Changing Chunk Lengths
To change the chunk length, simply change the `CHUNK_LENGTH` variable in the `config.py` file. Then, remove the directory `datasets/[dataset]/[dataset]_chunked/`, and run `create_chunked_data()` from `training_data_processing.py`

## Changing Training Parameters
Training parameters such as batch size and learning rate can be passed as command line arguments to the `model_training.py` file. 

## Changing Models
To change models, it should be sufficient to edit the `model.py` file. As long as your model has the same name, input size, and output size it should work out of the box. If the model name is changed, then the code in `model_training.py` must be changed to call the new name.

