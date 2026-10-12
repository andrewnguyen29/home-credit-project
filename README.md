# Home Credit Default Risk

Project workspace for exploring the Home Credit data, engineering features, training models, and generating submissions.

## Directory layout

- `data/raw/`: original source data; do not modify
- `data/interim/`: cleaned data and intermediate features
- `data/processed/`: model-ready datasets
- `docs/`: business and data documentation
- `notebooks/`: ordered exploratory and modeling notebooks
- `src/`: reusable R code (`data_preparation.R`)
- `models/`: saved model artifacts
- `reports/`: figures and evaluation metrics
- `submissions/`: generated submission files
- `tests/`: automated tests

## Data Preparation

`src/data_preparation.R` turns the findings from `notebooks/01_eda` (specifically its "Data preparation decisions" table) into reusable, leakage-safe cleaning and feature engineering functions. Every transformation in the script is tied back to a specific EDA finding in a comment above the relevant function — this was a deliberate choice so the preparation code stays auditable against the analysis that motivated it, rather than reading as a disconnected list of transformations.

**What it does:**
- Cleans the `DAYS_EMPLOYED` sentinel value (365243, ~18% of rows) into a flag plus a real tenure-in-years figure
- Converts `DAYS_BIRTH` into interpretable age in years
- Adds financial ratio features (credit-to-income, annuity-to-income, credit-to-goods-price)
- Adds a count of available `EXT_SOURCE` scores, motivated by its non-monotonic relationship with `TARGET` found in EDA
- Adds missing-data indicator flags for columns where missingness itself was correlated with `TARGET`
- Adds two interaction terms (age × employment tenure; debt-to-income × external-score count) and two binned features (age band, credit-income quintile)

**How train/test consistency is guaranteed:**
The script separates *fitting* from *applying*. `fit_preprocessing(train)` is the only function allowed to compute anything from data — it derives imputation medians and bin edges from `application_train` and returns them as a single stats object. `apply_preprocessing(df, stats, is_train)` takes that stats object and transforms either train or test identically; it is never allowed to compute a new statistic itself. This means the exact same median, the exact same bin edges, and the exact same set of output columns get used on both datasets — the mechanism the assignment calls "preventing data leakage."

`validate_train_test_consistency()` checks this directly: it confirms the prepared train and test datasets have identical columns (aside from `TARGET`, which only exists in train).

**Usage:** run the whole file from the repo root (with the two CSVs in `data/raw/`) and it defines the functions and then runs the pipeline in its last section: it prints the row and column counts and the consistency check, and saves the fitted statistics and prepared data to git-ignored folders. To load only the functions (as the modeling notebook does), set `SKIP_PIPELINE <- TRUE` before `source()`. The same steps step by step:
```r
SKIP_PIPELINE <- TRUE
source("src/data_preparation.R")

train <- fread("data/raw/application_train.csv")
test  <- fread("data/raw/application_test.csv")

prep_stats  <- fit_preprocessing(train)
train_ready <- apply_preprocessing(train, prep_stats, is_train = TRUE)
test_ready  <- apply_preprocessing(test,  prep_stats, is_train = FALSE)

check <- validate_train_test_consistency(train_ready, test_ready)
stopifnot(check$identical)

saveRDS(prep_stats, "models/preprocessing_stats.rds")
```

**Scope note:** the EDA notebook only explored `application_train`/`application_test`; the supplementary tables (bureau, previous applications, POS/cash and credit-card balances, installment payments) were optional at that stage and were not analyzed. This script's cleaning and feature engineering is therefore limited to the application-level data as well. A commented-out stub at the end of the script shows where supplementary-table aggregation and joining would be added if that analysis is done later — the pattern (aggregate to one row per `SK_ID_CURR`, then left-join) is noted but not implemented, since implementing it without the corresponding EDA would mean guessing at decisions the analysis hasn't actually made yet.


**Fix made during modeling:** blank strings in character columns (for example `OCCUPATION_TYPE`) were read as `""` rather than `NA`, so `OCCUPATION_TYPE_MISSING` was always `FALSE`. `add_missing_indicators()` now treats blank strings as missing. Use the updated `data_preparation.R`.

## Modeling

`notebooks/modeling.qmd` (rendered to `modeling.html`) builds, compares and tunes models for the default-risk target, then produces the Kaggle submission. The modeling functions are defined inside the notebook (in a collapsed code block), and the cleaning and feature engineering functions are in `src/data_preparation.R`. Preparation statistics and category levels are re-fit inside every cross-validation fold on training rows only, and any resampling is applied to training rows only.

**Models tried** (3-fold stratified CV, 40,000-row sample, metric = AUC):

| Model | CV AUC |
|---|---|
| Majority class (benchmark) | 0.5000 (accuracy 91.93%) |
| Logistic regression, 3 external scores | 0.7151 |
| Elastic-net logistic, raw columns | 0.7321 |
| Elastic-net logistic, engineered features | 0.7395 |
| Random forest, depth 8 / depth 14 | 0.7245 / 0.7199 |
| Gradient boosting, raw columns | 0.7410 |
| Gradient boosting, engineered features (default settings) | 0.7415 |
| Gradient boosting, tuned structure, 600 trees | **0.7425** |

**Class imbalance.** Class weights, downsampling, upsampling and SMOTE (numeric columns only) were compared with no adjustment. None improved gradient boosting (best adjusted: 0.7393 vs 0.7415 unadjusted), and the differences for the elastic net were within noise. No adjustment was used.

**Tuning.** A 20-iteration randomized search on a 10,000-row sample (3-fold CV) over learning rate, tree depth, leaf size, row sampling and tree count. The winning settings were re-checked on the larger folds, and the tree count was re-selected there (600 trees). The final model was trained on all 307,511 training rows.

**Why this model was selected.** Gradient boosting on the engineered features with the tuned structure has the highest mean CV AUC and the flattest curve as trees are added. The margin over the default settings (0.001) and over the elastic-net logistic regression (0.003) is smaller than the fold-to-fold standard deviation (about 0.01), so the choice is a judgment call rather than a clear win. The three external scores account for about 68% of the final model's relative influence.

**Sensitivity check.** Removing gender, family status, children and family-size fields lowers CV AUC by only 0.0013 (0.7412), so the model does not depend on them.

**Supplementary tables** were not used (not available in this environment).

**Kaggle score:** public leaderboard AUC of **0.74224**, compared to a cross-validated estimate of 0.7425. The two agree closely, which suggests the cross-validation was not overly optimistic.

**To reproduce:**
```r
# from the repo root, with data in data/raw/
quarto render notebooks/modeling.qmd
```
The full run takes roughly 45 minutes on 2 cores; slow steps are cached under `models/cache/` (git-ignored). `MODELING_SMOKE=1` runs a tiny version for testing. Packages: `data.table`, `ggplot2`, `glmnet`, `ranger`, `gbm`, `themis`, `recipes`.
