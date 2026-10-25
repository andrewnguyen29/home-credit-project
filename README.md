# Home Credit Default Risk

This project uses the Home Credit data to explore the applications, build features, train models, and create a Kaggle submission.

## Directory layout

- `data/raw/`: original source data; do not modify
- `data/interim/`: cleaned data and intermediate features
- `data/processed/`: model-ready datasets
- `docs/`: business and data documentation
- `notebooks/`: exploratory, modeling and model card notebooks
- `src/`: reusable R code (`data_preparation.R`)
- `models/`: saved model files
- `reports/`: figures and evaluation metrics
- `submissions/`: generated submission files
- `tests/`: automated tests

## Data Preparation

`src/data_preparation.R` turns the findings from my EDA notebook into reusable functions. It follows the "Data preparation decisions" table in that notebook. Each function has a comment that points back to the EDA finding behind it, so the code can be checked against the analysis.

**What it does:**
- Flags the impossible `DAYS_EMPLOYED` code (365243, about 18% of rows) and converts the valid values into years employed
- Converts `DAYS_BIRTH` into age in years
- Adds three ratios: credit to income, annuity to income, and credit to goods price
- Adds a count of available `EXT_SOURCE` scores. The EDA showed that the count relates to `TARGET`
- Adds missing-value flags for columns where the missing rate differed by `TARGET`
- Adds two interaction terms (age by years employed, annuity-to-income by score count) and two binned features (age band and credit-to-income quintile)

**How train and test stay consistent:**
The script separates fitting from applying. `fit_preprocessing(train)` is the only function that computes anything from the data. It calculates the imputation medians and bin cut points from `application_train` only and returns them as one list. `apply_preprocessing(df, stats, is_train)` uses that list on either train or test. It never calculates a new statistic. This means both datasets get the same medians, the same bin cut points, and the same columns. This is how the script prevents data leakage.

`validate_train_test_consistency()` checks the result. It confirms that the prepared train and test data have the same columns, except for `TARGET`, which only exists in train.

**How to run it:** run the whole file from the repo root, with the two CSV files in `data/raw/`. It defines the functions and then runs the pipeline in the last section. It prints the row and column counts and the consistency check. It also saves the fitted statistics and the prepared data to folders that Git ignores. To load only the functions, as the modeling notebook does, set `SKIP_PIPELINE <- TRUE` before `source()`. The same steps one at a time:
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

**Scope:** my EDA only covered `application_train` and `application_test`. I did not analyze the supplementary tables (bureau, previous applications, POS cash and credit card balances, installment payments), so the script covers the application data only. A commented-out example at the end shows where the supplementary tables would go. The pattern is to summarize each table to one row per `SK_ID_CURR` and then left-join it. I did not build it because the EDA did not make those decisions.

**Fix made during modeling:** `fread()` reads blank text fields as `""` and not `NA`. This made `OCCUPATION_TYPE_MISSING` `FALSE` for every row, even though 96,391 applicants have a blank occupation. `add_missing_indicators()` now counts blank strings as missing. Use the updated `data_preparation.R`.

## Modeling

`notebooks/modeling.qmd` (compiled to `modeling.html`) builds, compares, and tunes the models and creates the Kaggle submission. The modeling functions are in a collapsed code block inside the notebook. The cleaning and feature functions come from `src/data_preparation.R`. In every cross-validation fold, the preparation statistics are re-fit on the training rows only. Any resampling also touches the training rows only.

**Models tried** (3-fold cross-validation on a 40,000-row sample, scored by AUC):

| Model | CV AUC |
|---|---|
| Majority class (benchmark) | 0.5000 (accuracy 91.93%) |
| Logistic regression, 3 external scores | 0.7151 |
| Elastic-net logistic, raw columns | 0.7321 |
| Elastic-net logistic, engineered features | 0.7395 |
| Random forest, depth 8 / depth 14 | 0.7245 / 0.7199 |
| Gradient boosting, raw columns | 0.7410 |
| Gradient boosting, engineered features (default settings) | 0.7415 |
| Gradient boosting, tuned settings, 600 trees | **0.7425** |

**Class imbalance:** I compared class weights, downsampling, upsampling, and SMOTE (numeric columns only) to no adjustment. None improved gradient boosting. The best adjusted result was 0.7393, compared to 0.7415 with no adjustment. The elastic-net differences were within the noise between folds. I used no adjustment.

**Tuning:** I ran a random search of 20 settings on a 10,000-row sample with 3-fold cross-validation. It covered learning rate, tree depth, leaf size, row sampling, and number of trees. I then checked the best settings on the larger folds and re-selected the tree count there, which gave 600 trees. The final model was trained on all 307,511 training rows.

**Why I chose this model:** gradient boosting on the engineered features with the tuned settings has the highest mean CV AUC. Its AUC also stays flat as trees are added, compared to the default settings, which get worse after 300 trees. The gap is small. It is 0.001 above the default settings and 0.003 above the elastic-net logistic regression, and the fold-to-fold standard deviation is about 0.01. This suggests that the choice is a judgment call and not a clear win. The three external scores make up about 68% of the final model's relative influence.

**Fairness check:** removing gender, family status, children, and family size lowers CV AUC by only 0.0013 (to 0.7412). This suggests that the model does not depend on those fields.

**Supplementary tables:** not used. They were not available in my environment.

**Kaggle score:** the public leaderboard AUC is **0.74224**, compared to a cross-validated estimate of 0.7425. The two are close, which suggests that the cross-validation was not overly optimistic.

**To reproduce:**
```r
# from the repo root, with the data in data/raw/
quarto render notebooks/modeling.qmd
```
A full run takes about 45 minutes on 2 cores. The slow steps are saved under `models/cache/`, which Git ignores. Setting `MODELING_SMOKE=1` runs a tiny version for testing. Packages needed: `data.table`, `ggplot2`, `glmnet`, `ranger`, `gbm`, `themis`, `recipes`.

## Model Card

`notebooks/model_card.qmd` (compiled to `model_card.html`) documents the final model for a business reader who is deciding whether to deploy it. The code is hidden in the HTML, so it reads like a document. It covers the sections the assignment asks for:

- **Model details and intended use:** gradient boosting (600 trees, depth 3, 135 inputs) trained on 307,511 applications, who should use it, and what it should not be used for.
- **Performance:** AUC is 0.7425 in cross-validation, 0.74224 on Kaggle, and 0.7535 on a 20% holdout. At the chosen threshold, precision is 33.7% and recall is 20.4% on the holdout.
- **Decision threshold:** I used published figures for the loss on a problem loan (60% loss given default on 70% of the credit, so 42%) and the margin on a repaid loan (14%, from Home Credit's 15.7% net interest margin). The break-even predicted risk is 25%, and the best threshold on the holdout is **24.0%**. At that threshold the model is worth about **120M per 100,000 applications** compared to approving everyone (95% range 93M to 146M). A sensitivity table shows how the best threshold and the benefit change with the cost assumptions. The benefit is positive in 8 of the 9 scenarios I tested.
- **Explainability:** SHAP values on a 1,000-row sample. The three external credit scores are the top inputs.
- **Adverse action mapping:** plain-language denial reasons for the top inputs, and an example applicant. Gender, education and age are not given as reasons.
- **Fairness:** approval and flag rates by gender and education at the chosen threshold. Men are flagged 2.2 times as often as women, and the gap follows their different problem rates. I recommend removing gender before deployment.
- **Limitations and an executive summary.**

**Note on the saved model:** `modeling.qmd` saved only the test predictions, not the fitted model. The model card fits the final model (all rows) and an evaluation model (80% of rows) once and saves them in `models/cache/`, which Git ignores. It uses the same settings and the same functions as `modeling.qmd`, which it loads from that file. I estimated SHAP values with a sampling method that I wrote, because the SHAP packages could not be installed for my version of R.

**To reproduce:**
```r
# from the repo root, with the data in data/raw/ and after running modeling.qmd once
quarto render notebooks/model_card.qmd
```
The first run takes about 40 minutes (two model fits and the SHAP step). Later runs take under a minute. Packages needed: `data.table`, `ggplot2`, `gbm`, `knitr`, `scales`.
