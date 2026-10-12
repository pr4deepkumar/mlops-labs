## Changes made for the Lab2 assignment

### `src/train_model.py`
- Simplified the folder setup for `data/` and `models/`. The old code repeated the same block twice, once for when the folder existed and once for when it didn't. Now it just creates the folder if it's missing.
- Switched the file paths to `pathlib`.
- Removed unused imports and commented-out code.
- Logged more to MLflow: the random seed, the model's settings, and the trained model itself.

### `src/evaluate_model.py`
- Rewrote the error messages so it's easier to tell what went wrong when the model file is missing or the data can't be generated.
- Fixed a bug where the evaluation data could end up with 0 samples and crash the script.
- Added accuracy, precision, recall and ROC-AUC to the saved metrics, next to the F1 score.
- Removed unused imports and a folder creation step the script didn't need.

### `Lab2/workflows/model_calibration.yml`
- Fixed a bug where the nightly run committed the model but skipped the metrics file without any warning.
- Fixed the "Ghithub" typo in the workflow name.

### `Lab2/workflows/model_calibration_on_push.yml`
- Retraining now runs only when something in Lab2 changes, not on every push to `main`.

### Both `Lab2/workflows`
- Upgraded `actions/setup-python` from v4 to v5.
- Upgraded Python from 3.9 to 3.11.
- Added `cache: pip` to speed up dependency installs.
- Set Lab2 as the working folder, so the commands no longer need the full path.
- Removed the file permission steps. The script runs through `python`, so it doesn't need them.
- Added the option to run the workflow manually from the Actions tab.
- Stopped the nightly and on-push runs from pushing at the same time.
- Each run also uploads the model and metrics as a downloadable artifact.
- Commits now come from the GitHub Actions bot instead of a personal account.

### `.github/workflows/` (repo root)
- Applied the same changes to the root copies of both workflows, since GitHub Actions only runs workflows from the root `.github/workflows/` directory.

The model now retrains whenever Lab2 changes on `main`, every night, and whenever triggered by hand, and every run saves both the model and its metrics.

# Using GitHub Actions for Model Training and Versioning

This repository demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows you to easily update and improve your model in a collaborative environment.

Watch the tutorial video for this lab at [Github action Lab2](https://youtu.be/cj5sXIMZUjQ)


## Prerequisites

- [GitHub](https://github.com) account
- Basic knowledge of Python and machine learning
- Git command-line tool (optional)

## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
3. **Clone Your Repository**:
   ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo

   ```
4. GitHub account
5. Basic knowledge of Python and machine learning
6. Git command-line tool (optional)

# Running the Workflow
## Customize Model Training
1. Modify the `train_model.py` script in the `src/` directory according to your dataset and model requirements. This script generates synthetic data for demonstration purposes.

## Push Your Changes:
1. Commit your changes and push them to your forked repository.

## GitHub Actions Workflow:
1. Once you push changes to the main branch, the GitHub Actions workflow will be triggered automatically.

## View Workflow Progress:
1. You can track the progress of the workflow by going to the "Actions" tab in your GitHub repository.

## Retrieve the Trained Model:
1. After the workflow completes successfully, the trained model will be stored in the `models/` directory.

# Model Evaluation
The model evaluation is performed automatically within the GitHub Actions workflow. The evaluation results (e.g., F1 Score) are stored in the `metrics/` directory.

# Versioning the Model
Each time you run the workflow, a new version of the model is created and stored. You can access and use these models for your projects.

# GitHub Actions Workflow Details
The workflow consists of the following steps:

- Generate and Store Timestamp: A timestamp is generated and stored in a file for versioning.
- Model Training: The `train_model.py` script is executed, which trains a random forest classifier on synthetic data and stores the model in the `models/` directory.
- Model Evaluation: The `evaluate_model.py` script is executed to evaluate the model's F1 Score on synthetic data, and the results are stored in the `metrics/` directory.
- Store and Version the New Model: The trained model is moved to the `models/` directory with a timestamp-based version.
- Commit and Push Changes: The metrics and updated model are committed to the repository, allowing you to track changes.

# Model Calibration Workflow
## Overview
The `model_calibration_on_push.yml` workflow is a part of the automation process for machine learning model calibration within this repository. It is essential for ensuring that the model's predictions are accurate and well-calibrated, a critical step in machine learning applications.

## Workflow Purpose
This workflow's primary purpose is to calibrate a trained machine learning model after each push to the main branch of the repository. Calibration is a crucial step to align model predictions with reality, particularly when dealing with classification tasks. In simple terms, calibration ensures that a model's predicted probabilities match the actual likelihood of an event happening.

## Workflow Execution
Let's break down how this workflow operates step by step:

### Step 1: Trigger on Push to Main Branch
- This workflow is automatically initiated when changes are pushed to the main branch of the repository. It ensures that the model remains calibrated and up-to-date with the latest data and adjustments.

### Step 2: Prepare Environment
- The workflow begins by setting up a Python environment and installing the necessary Python libraries and dependencies. This is crucial to ensure that the model calibration process can be executed without any issues.

### Step 3: Load Trained Model
- The trained machine learning model, which has been previously saved in the `models/` directory, is loaded into memory. This model should be the most recent version, as trained by the `train_model.py` script.

### Step 4: Calibrate Model Probabilities
- In this step, the model's predicted probabilities are calibrated. Calibration methods, such as Platt scaling or isotonic regression, are applied. These methods adjust the model's predicted probabilities to match the actual likelihood of an event occurring. This calibration step is critical for reliable decision-making based on the model's predictions.

### Step 5: Save Calibrated Model
- The calibrated model is saved back to the `models/` directory. It is given a distinct identifier to differentiate it from the original, uncalibrated models. This ensures that both the original model and the calibrated model are available for comparison and use.

### Step 6: Commit and Push Changes
- This final step involves committing the calibrated model and any other relevant files to the repository. It is essential to keep track of the changes made during the calibration process and to store the calibrated model in the repository for future applications and reference.

# Customization
The `model_calibration_on_push.yml` workflow can be customized to align with your specific project requirements. You can modify calibration methods, the directory where the calibrated model is saved, or any other aspects of the calibration process to meet your project's unique needs.

# Integration with Model Training
This workflow is designed to work seamlessly with the main model training workflow, `model_retraining_on_push.yml`. In the initial workflow, the model is trained, and in this workflow, the calibrated model is generated. The calibrated model can then be used in applications where precise, well-calibrated probabilities are essential.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- This project uses GitHub Actions for continuous integration and deployment.
- Model training and evaluation are powered by Python and scikit-learn.

# Questions or Issues
If you have any questions or encounter issues while using this GitHub Actions workflow, please open an issue in the Issues section of your repository.

