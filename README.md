# Photo-Z Challenge: Environment Setup and Execution Guide

<img src="spectra.png" alt="Photo-Z Challenge Overview" width="300"/>

![Photo-Z Challenge Overview](spectra.png)

Welcome to the Photo-Z Challenge! This guide provides step-by-step instructions to set up your Python environment, install all required dependencies, and run the machine learning pipeline on your local machine. It is fully compatible with both Windows and macOS.

## 1. Prerequisites

You must have the following tools installed on your system before beginning:

* **Git:** Required to download the repository and version control your changes. You can download and install it from the official Git website (https://git-scm.com/downloads).
* **Miniconda (Recommended) or Anaconda:** This is the most reliable way to manage Python environments and complex libraries across different operating systems. Download it from the official Anaconda website (https://docs.anaconda.com/free/miniconda/).

## 2. Directory Structure Setup

For the pipeline to run smoothly without modifying any paths in the code, you need to set up a specific folder structure. The datasets must be placed in a `data` folder located exactly one level above the code repository.

Your final workspace should look exactly like this:

```text
Workspace_Folder/
├── data/
│   ├── training_set.h5
│   ├── validation_set.h5
│   └── test_set.h5
└── photoz_challenge/
    ├── config.yaml
    ├── train_model.py
    └── ...
```

## 3. Download and Environment Setup

Open your terminal (macOS) or Anaconda Prompt (Windows) and execute the following commands in order.

**Step 3.1:** Clone the repository to your local machine.
```bash
git clone [https://github.com/gimarso/photoz_challenge.git](https://github.com/gimarso/photoz_challenge.git)
cd photoz_challenge
```

**Step 3.2:** Create the virtual environment using the provided configuration file. This will install all necessary dependencies (PyTorch, Pandas, Matplotlib, etc.).
```bash
conda env create -f environment.yml
```

**Step 3.3:** Activate the newly created environment. You must do this every time you open a new terminal to run the project.
```bash
conda activate photoz_env
```

## 4. Running the Pipeline (Command Line)

Once your environment is active and your data is properly placed in the `../data/` directory, you can run the different stages of the pipeline sequentially using the provided Python scripts:

* **Train the Model:**
  Trains the neural network using `training_set.h5` and saves the model weights in the `../models/` directory according to your `config.yaml`.
  ```bash
  python train_model.py
  ```

* **Evaluate on Validation Set:**
  Loads the trained model, runs inference on `validation_set.h5`, and generates detailed evaluation plots (PDFs) in the `../pdf/` directory.
  ```bash
  python test_validation.py
  ```

* **Evaluate on Test Set:**
  Similar to the validation step, but runs inference and generates evaluation metrics specifically for the `test_set.h5`.
  ```bash
  python test_test.py
  ```

* **Visualize Datasets:**
  Generates diagnostic plots (Redshift distribution, Color-Magnitude, etc.) for a specific dataset. By default, it looks at the validation set, but you can pass any file path.
  ```bash
  python visualize_data.py --file ../data/validation_set.h5
  ```

## 5. Running the Pipeline (Jupyter Notebook)

Alternatively, you can run and interact with the pipeline using a Jupyter Notebook. This approach is highly recommended for exploring the data interactively, debugging, and visualizing results step-by-step.

**Step 5.1:** Launch Jupyter from your terminal (make sure the `photoz_env` environment is activated first):
```bash
jupyter notebook
```
*(Note: You can also use `jupyter lab` if you prefer the modern interface).*

**Step 5.2:** Your default web browser will automatically open. Navigate through the directory tree, open the `.ipynb` notebook file included in the repository, and run the cells sequentially to execute the pipeline.