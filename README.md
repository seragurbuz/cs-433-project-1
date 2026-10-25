# CS-433 Project 1

Predict MICHD labels (`-1` or `1`) from the supplied BRFSS dataset. The six required
algorithms are implemented in `implementations.py` using NumPy.

## Setup

Use Python 3.9 or newer and NumPy.
Download the dataset from the course's AIcrowd competition and extract it into
`dataset/`, containing `x_train.csv`, `x_test.csv`, and `y_train.csv` with their
original names and IDs.
Install Matplotlib if you want the optional cross-validation diagnostic plots.

## Reproduce predictions

```sh
python run.py
```

This validates or rebuilds cleaned data, selects a threshold, evaluates on a
reserved holdout, refits on all labeled rows, and writes `submission.csv` to the
repository root. `python trainer.py` runs the same workflow. Full training and
five-fold CV can take several minutes or more and require several GB of RAM.
Paths resolve relative to the source files, including when invoked from elsewhere.

Defaults in `trainer.py`: logistic regression, 1,000 gradient steps, learning rate
0.1, 20% stratified holdout, five inner folds, seed 42. `lambda_ = 0.01` applies
only when selecting a regularized method. Changing settings invalidates the
cached threshold. Least squares requires a nonsingular design matrix; redundant
features may require additional feature selection before using that method.

## Preprocessing

`feature_rules.py` selects 250 predictors in codebook order, excluding identifiers,
weights, redundant fields and potential leakage fields. `preprocessing.py` uses
`codebook_spec.json` to convert missing/refused/inapplicable codes to NaN, decode
frequencies and durations, convert heights and weights, and reject invalid values.
These fixed rules do not estimate statistics from the data. No additional feature
columns are generated; categorical codes are currently treated numerically.

`prepare_data.py` writes feature-only CSVs under `dataset/cleaned_data/`, preserving
row order. All training entry points validate both headers and a manifest of
SHA-256 hashes covering raw inputs, preprocessing code and cleaned outputs.
Old 31-feature exports, changed rules, edited files or missing manifests cause
regeneration.

To force regeneration:

```sh
python prepare_data.py --data dataset
```

Missing values are imputed with training-column medians (zero for entirely missing
columns). Scaling uses training means and standard deviations; constant columns
use a standard deviation of one. A bias column is added. Validation and test rows
never contribute to fitting these statistics.

## Threshold selection and evaluation

The stratified outer holdout is reserved first. Five-fold stratified CV uses only
the remaining 80% of rows, fitting imputation and scaling separately within each
fold. The raw-score threshold maximizing positive-class F1 is selected from pooled
out-of-fold scores; tied scores are kept together.

The model is fitted on the outer training portion and evaluated once on the
untouched holdout at that threshold. Holdout F1, accuracy, precision and recall
are recorded. Finally, preprocessing and the model are refitted on all labeled
rows, retaining the chosen threshold, to generate test predictions.

For threshold selection alone, without evaluating the holdout or making a submission:

```sh
python experiments.py
```

This shares the trainer's split and settings. Compatible selections are reused;
metadata includes the training split hash, preprocessing manifest, settings and
tuning code hashes. The out-of-fold F1 is a tuning score, not an unbiased final
performance estimate.
