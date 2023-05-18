## Description
This repo is meant to facilitate model training for the Acoustic Species ID project, and allow for rapid, iterative testing of different data and models.

## Usage:
To train the current model (efficientnet4), simply run the model_training.py file. 

## Dataset Conventions
Datasets are stored in the `datasets` folder. If `[dataset]` represents the name of the dataset, then paths to training data are in the following format: `datasets/[dataset]/[dataset]_[descriptor]/`. For example, one possible path is `datasets/cosmos/cosmos_chunked/`. Within every 
such folder, there will be several audio clips of training data, as well as a `labels.csv` file containing the corresponding labels for each file.

## Changing Datasets
To change the dataset, simply change the `DATASET` variable in the `config.py` file. If the path to the dataset does not already exist, you will have to create it, and place the data/labels in `datasets/[dataset]/[dataset]_original/`. You will also need to change `TRAINING_DATA_DIR` to correspond to the new data you want to train on. If this data does not exist yet, then you will have to create it as well (will be automated eventually!). To create chunked data, run `create_chunked_data()` from `training_data_processing.py`

## Changing Training Parameters
Training parameters such as batch size and learning rate can be passed as command line arguments to the `model_training.py` file. 

## Changing Models
To change models, it should be sufficient to edit the `model.py` file. As long as your model has the same name, input size, and output size it should work out of the box. If the model name is changed, then the code in `model_training.py` must be changed to call the new name.

## .vscode
This folder should contain at least a `launch.json` file that provides entry points into the package.  This could be entry points to example programs, typical debug entry points, etc.  Ideally, this will also contain a `settings.json` with folder specific VS Code settings.

## Python Packages
Almost all code should be organized into packages.  This facilitates easy code reuse.  See https://packaging.python.org/en/latest/ for more information on how to use packages in Python.

## tests
Tests should ideally be kept in an isolated folder.  This example uses `pytest`, see https://docs.pytest.org/en/7.0.x/ for information on how to use `pytest`.

## .gitignore
This is an important file to put into your `git` repositories.  This helps prevent `git` from including unnecessary files into the version control system, such as generated files, environment files, or log files.

## Jupyter Notebooks
Jupyter Notebooks are a great way to add interactive reports or documentation.  There is an example here.

## *.code-workspace
This is created by going to `File`->`Save Workspace As` in VS Code.  This allows us to easily open the same development environment across computers and operating systems.  Use this as the root of your development environment.

You'll also notice here that there are some recommended extensions.  This allows everyone on your team to have a consistent toolchain for how to interact with your code.  See https://code.visualstudio.com/docs/editor/extension-marketplace#_workspace-recommended-extensions for instructions for how to configure these.

## setup.py
This is the original way to package Python projects, which we probably still prefer.  However, in general, if you can, you should probably start using the newer packaging tools (`pyproject.toml`)

See https://packaging.python.org/en/latest/tutorials/packaging-projects/ and https://docs.python.org/3/distutils/setupscript.html for more information about each tool.

## Installing for developers
Developers should do the following:
```
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv/Scripts/activate.ps1   # for Windows PowerShell
.venv/Scripts/activate.bat   # for Windows Command Prompt
source .venv/bin/activate    # for bash

# Install in developer mode
python -m pip install -e .[dev]
```
