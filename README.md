# Home Credit Default Risk: Predicting Who Will Struggle to Repay

**In one paragraph:** Home Credit lends to people with little or no credit history, so it has very little to go on when it decides who to approve. I built a gradient boosting model, in R, that ranks applicants by their risk of payment difficulties. The riskiest tenth of applicants has a problem rate of 27.0%, compared to 1.5% for the safest tenth. The model scores 0.742 AUC on Kaggle. Under cost assumptions taken from published sources, sending the riskiest ~5% of applicants to manual review is worth about **120M per 100,000 applications**. I documented the model for a business reader in a model card that covers the decision threshold, explanations for denials, fairness, and limits.

![Risk by group](reports/figures/risk_deciles.png)

## At a glance

| | |
|---|---|
| **Data** | 307,511 loan applications from the Kaggle Home Credit Default Risk competition, 8.07% with payment difficulties |
| **Model** | Gradient boosting (R `gbm`), 600 trees, 135 inputs including 17 features I built |
| **Kaggle score** | **0.74224 AUC** (public leaderboard). Cross-validated: 0.7425. 20% holdout: 0.7535 |
| **Chosen threshold** | Flag applicants with predicted risk of 24% or more (4.9% of applicants) |
| **At that threshold** | Flagged group has a 33.7% problem rate (vs. 8.1% overall). The flag catches 20.4% of all problem loans |
| **Business value** | About 120M per 100,000 applications (95% range 93M to 146M), in dataset currency units, before review costs |
| **Biggest driver** | Three external credit scores make up 68% of the model's relative influence |

## The business problem

Home Credit makes installment loans to people that banks often turn away. Many of these applicants have no credit bureau record. This leads to two costly mistakes: approving loans that go bad, and declining people who would have repaid. Losses cut profit, and good applicants who are declined may end up with worse lenders.

**Objective:** score every applicant at the time of application, so that underwriters can spend their review effort on the few who need it, without turning away good customers. I measured success with AUC, which shows how well a model ranks the people who struggle above the people who repay. The full problem statement is in [`docs/Home_Credit_Business_Problem_Statement.docx`](docs/Home_Credit_Business_Problem_Statement.docx).

## My solution

1. **Explored the data** ([EDA notebook](notebooks/01_eda.qmd)). I found that about 18% of applicants have an impossible job length (a placeholder code, about 1,000 years), that the external credit scores are often missing, and that missing scores relate to the outcome.
2. **Prepared the data** ([`src/data_preparation.R`](src/data_preparation.R)). I turned each EDA finding into a reusable function: cleaned job length, age in years, credit-to-income and credit-to-price ratios, a count of available scores, and missing-data flags. All statistics are learned on training rows only, so nothing leaks from test data.
3. **Compared and tuned models** ([modeling notebook](notebooks/modeling.qmd)). I tested logistic regression, elastic net, random forest, and gradient boosting, with leakage-safe 3-fold cross-validation. I tried four fixes for class imbalance and ran a randomized search for the final settings.
4. **Turned the model into a decision** ([model card](notebooks/model_card.qmd)). I chose a threshold from published loan cost figures, explained the model with SHAP values, translated its drivers into plain-language denial reasons, and checked fairness by gender and education.

| Model (3-fold CV on 40,000 rows) | AUC |
|---|---|
| Majority class benchmark | 0.5000 |
| Logistic regression, 3 external scores | 0.7151 |
| Random forest | 0.7245 |
| Elastic net, engineered features | 0.7395 |
| Gradient boosting, default settings | 0.7415 |
| **Gradient boosting, tuned (final)** | **0.7425** |

The final choice is only 0.001 above the default settings and 0.003 above the elastic net. Fold-to-fold noise is about 0.01. This suggests that the choice is a judgment call and not a clear win.

## The business value

- **It finds the risk.** On 61,502 applicants the model had never seen, the riskiest tenth had a 27.0% problem rate and the safest tenth had 1.5%. One third of all problem loans sit in the riskiest tenth.
- **A small review policy pays off.** At the 24% threshold, the flagged group (4.9% of applicants) holds 20.4% of the problem loans. The problem rate among approved applicants falls from 8.1% to 6.8%.
- **The money.** I used published recovery studies and Home Credit's own 2018 financial report for the loss on a problem loan (42% of credit) and the margin on a repaid loan (14% of credit). Compared to approving everyone, the policy is worth about **120M per 100,000 applications** (95% range 93M to 146M). The break-even risk is 25%, and the best threshold is 24%.
- **The caveat.** The benefit is positive in 8 of the 9 cost scenarios I tested, but its size changes a lot with the assumptions. It turns negative when the loss is low and the margin is high. Home Credit should replace my assumptions with its own figures. The cost of running the reviews is not included.

![Net benefit by threshold](reports/figures/threshold_net_benefit.png)

## Difficulties I ran into

- **Data leakage is easy to cause.** Medians, bin cut points, and category levels all come from data. I split fitting from applying, and I re-fit every statistic inside each cross-validation fold.
- **A silent bug.** `fread()` reads blank text as `""` and not `NA`. My `OCCUPATION_TYPE` missing flag was `FALSE` for every row, even though 96,391 applicants have a blank occupation. I caught it during modeling, fixed the function, and reran everything.
- **Tuning did not help at first.** My tuned settings did not beat the defaults at 200 trees. The best settings needed more trees than my search allowed. Re-selecting the tree count on larger folds gave the final 600 trees.
- **Positron would not fork R.** My parallel code failed. I rewrote it with fresh R processes (a PSOCK cluster) and fixed a missing-constants error on the workers.
- **The model was not saved.** My modeling notebook saved only the test predictions. For the model card I had to refit the model and keep the fitted objects. The SHAP packages also would not install for my version of R, so I wrote a sampling-based SHAP estimator and checked that its values add up to the predictions.
- **The data has no costs.** Loss and margin are not in the dataset. I had to find real figures, and I had to be clear about which numbers are published and which are my assumptions.
- **The label is not a final loss.** `TARGET` flags a late payment on an early installment, so a flagged loan may still have been repaid.

## What I learned

- **Ranking is not a decision.** An AUC of 0.74 says little about money until you choose a threshold and attach costs. The threshold decides whether the model gains 120M or loses hundreds of millions.
- **Honest validation takes effort.** The Kaggle score (0.742) was close to my cross-validated estimate (0.7425), and I can trust that because I kept preparation inside the folds.
- **Small gains need small claims.** The gap between my best models was inside the noise, so I reported the choice as a judgment call.
- **A model card changes how you think.** Writing denial reasons, fairness tables, and limits made me look at inputs I would normally ignore. Gender is a model input, and the fairness tables showed men are flagged 2.2 times as often as women. I now recommend removing it.
- **Presenting the results.** I turned this work into a 9-minute presentation for decision-makers. Leading with the business result and putting the process in an appendix made the same findings much easier to follow.

## Interview talking points

- **Why AUC?** About 92% of applicants repay, so accuracy is misleading (a model that approves everyone is 92% accurate). AUC measures ranking, which is what an underwriter needs.
- **How did I avoid leakage?** Fit and apply are separate functions. Every fold re-fits medians, bin edges, and category levels on its own training rows. Resampling touches training rows only.
- **Why gradient boosting?** It had the highest CV AUC, and its AUC stayed flat as I added trees. I would say plainly that the gap to the elastic net was small.
- **How did I pick the threshold?** Approve when expected margin is above expected loss, so the break-even risk is margin / (margin + loss) = 25%. I checked it against every threshold on a holdout and picked 24%.
- **What would I do about fairness?** Remove gender, repeat the fairness tables, and have lawyers review education and age. The flag is equally accurate for men and women (33% and 34% problem rate among flagged applicants), but the flag rates differ.
- **What are the biggest risks?** Dependence on three outside scores, a label that is not a final loss, approved-applicants-only data, and cost assumptions that are not Home Credit's.
- **What would I do next?** Add the supplementary tables (credit bureau, previous applications, installments, credit cards, POS cash), run a pilot to replace my cost assumptions, and monitor AUC and fairness monthly.

## What is in this repository

| Item | What it is |
|---|---|
| [`notebooks/01_eda.qmd`](notebooks/01_eda.qmd) ([PDF](notebooks/pdf/01_eda.pdf)) | Exploratory analysis and data preparation decisions |
| [`src/data_preparation.R`](src/data_preparation.R) | Reusable cleaning and feature functions, with train and test checks |
| [`notebooks/modeling.qmd`](notebooks/modeling.qmd) ([PDF](notebooks/pdf/modeling.pdf)) | Model comparison, class imbalance, tuning, final model, Kaggle score |
| [`notebooks/model_card.qmd`](notebooks/model_card.qmd) ([PDF](notebooks/pdf/model_card.pdf)) | Model card: threshold, SHAP, denial reasons, fairness, limits, executive summary |
| [`reports/figures/`](reports/figures) | Figures used in this README |
| [`docs/`](docs) | Business problem statement |

GitHub shows `.qmd` files as source code, so I added a PDF of each rendered notebook. Data, saved models, and submission files are not committed (see `.gitignore`).

## Model card summary

The model card is written for a business reader who is deciding whether to deploy the model.

- **Intended use:** rank applicants and send high-risk ones to a human review. It is not meant for automatic denial, pricing, or other lenders and products.
- **Explainability:** SHAP values on a 1,000-applicant sample. The three outside scores lead, followed by loan-to-price ratio, annuity, and employment history.
- **Denial reasons:** each top input has a plain-language reason. Gender, education, and age are never given as reasons.
- **Fairness:** flag rates and impact ratios by gender and education at the chosen threshold, with a recommendation to remove gender.
- **Limitations:** the label is not a final loss, the data covers approved applicants only, the model uses only the main application table, and the cost figures are assumptions.

![Top SHAP inputs](reports/figures/shap_top15.png)

## Reproducing my work

Put `application_train.csv` and `application_test.csv` in `data/raw/`, open the repo folder as the working directory, and run these from the repo root:

```r
source("src/data_preparation.R")             # runs the preparation pipeline
quarto render notebooks/modeling.qmd         # about 45 minutes on 2 cores, cached
quarto render notebooks/model_card.qmd       # about 40 minutes the first time, then under a minute
```

Packages: `data.table`, `ggplot2`, `glmnet`, `ranger`, `gbm`, `themis`, `recipes`, `knitr`, `scales`. Setting `MODELING_SMOKE=1` runs a tiny version of the modeling notebook for testing.

**Author:** Andrew Nguyen, MSBA, University of Utah
