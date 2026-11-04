# Heart Disease Classification — Educational Study

A small applied machine-learning study comparing three classifiers on the Cleveland heart-disease dataset. The repository includes the cleaned and original tabular data, exploratory-analysis scripts, model-evaluation code, and saved result figures.

## Models and reported results

The archived results report test accuracy of 86.9% for Random Forest, 83.6% for Support Vector Machine, and 77.0% for Multilayer Perceptron. These are results from the supplied experiment and should not be interpreted as clinical performance guarantees.

## Repository layout

```text
data/       source and cleaned CSV files
src/        exploratory analysis and training/evaluation scripts
output/     saved figures and comparison results
requirements.txt
```

## Setup and use

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python src/eda.py
python src/train_eval.py
```

Scripts in this curated copy use paths relative to the repository instead of machine-specific absolute paths.

## Data provenance and responsible use

The original project identifies the UCI Heart Disease Repository (Cleveland) as its data source. Confirm the current citation and reuse terms before public redistribution. This is an educational exercise only—not a medical device, diagnostic tool, or substitute for professional clinical judgment.

## Publication note

The supplied source was not executed as part of the archive-sorting task. Verify dependency versions, reproduce the reported metrics, and review the dataset license before publishing.