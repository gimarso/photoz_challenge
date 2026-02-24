# Photo-Z Challenge: Environment Setup and Execution Guide

![Photo-Z Challenge Overview](link_to_your_image_here.png)

Welcome to the Photo-Z Challenge! This guide provides step-by-step instructions to set up your Python environment, install all required dependencies, and run the machine learning pipeline on your local machine. It is fully compatible with both Windows and macOS.

## 1. Prerequisites

You must have the following tools installed on your system before beginning:

* **Git:** Required to download the repository and version control your changes. You can download and install it from the official Git website (https://git-scm.com/downloads).
* **Miniconda (Recommended) or Anaconda:** This is the most reliable way to manage Python environments and complex libraries across different operating systems. Download it from the official Anaconda website (https://docs.anaconda.com/free/miniconda/).

## 2. Directory Structure Setup

For the pipeline to run smoothly without modifying any paths in the code, you need to set up a specific folder structure. The datasets must be placed in a `data` folder located exactly one level above the code repository.

Your final workspace should look exactly like this:

`Workspace_Folder/`
`├── data/`
`│   ├── training_set.h5`
`│   └── validation_set.h5`
`└── photoz_challenge/`
`    ├── config.yaml`
`    ├── train_model.py`
`    └── ...`

## 3. Download and Environment Setup

Open your terminal (macOS) or Anaconda Prompt (Windows) and execute the following commands in order.

**Step 3.1:** Clone the repository to your local machine.
```bash
git clone [https://github.com/gimarso/photoz_challenge.git](https://github.com/gimarso/photoz_challenge.git)
cd photoz_challenge