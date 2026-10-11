# =============================================================================
# data_preparation.R
#
# Purpose:
#   Turn the exploratory findings from notebooks/01_eda (see "Data preparation
#   decisions" table) into reusable, leakage-safe cleaning and feature
#   engineering functions for the Home Credit Default Risk project.
#
# Design contract:
#   - Every statistic used to transform the data (medians, bin edges, etc.)
#     is computed ONCE from application_train, via fit_preprocessing().
#   - Those statistics are stored in a single list object and passed into
#     apply_preprocessing() for BOTH train and test.
#   - No function in this file ever computes a statistic from test data.
#     This is the mechanism that prevents validation/test information from
#     leaking into how the training data itself is prepared.
#
# Usage:
#   Run this whole file from the repo root (data in data/raw/) and it runs the
#   pipeline at the bottom (Section 7). Or, to use only the functions:
#   SKIP_PIPELINE <- TRUE
#   source("src/data_preparation.R")
#   train <- fread("data/raw/application_train.csv")
#   test  <- fread("data/raw/application_test.csv")
#
#   prep_stats   <- fit_preprocessing(train)
#   train_ready  <- apply_preprocessing(train, prep_stats, is_train = TRUE)
#   test_ready   <- apply_preprocessing(test,  prep_stats, is_train = FALSE)
#
#   check <- validate_train_test_consistency(train_ready, test_ready)
#   stopifnot(check$identical)
#
#   saveRDS(prep_stats, "models/preprocessing_stats.rds")  # for reuse later
#
# Scope note:
#   The EDA notebook explicitly limited itself to application_train/test
#   (the supplementary tables — bureau, previous_application, POS_CASH,
#   credit_card, installments — were optional and not explored). This script
#   mirrors that scope: application-level cleaning and feature engineering
#   only. A stub section at the end shows where supplementary-table
#   aggregation would plug in if that work is done later.
# =============================================================================

library(data.table)

# -----------------------------------------------------------------------------
# SECTION 1: Deterministic cleaning and feature engineering
#   These functions need no information from training data beyond what's in
#   the row itself, so they are safe to apply identically to train and test
#   with no "fitting" step.
# -----------------------------------------------------------------------------

clean_days_employed <- function(df) {
  # EDA decision (notebook section "Invalid codes and unusual values"):
  # DAYS_EMPLOYED == 365243 is an impossible sentinel value (about 1,000
  # years), present in 55,374 rows / 18.01% of application_train. Decision:
  # do not delete these rows or silently impute the sentinel away — flag it
  # and convert genuinely negative day-offsets into a real "years employed"
  # figure, matching the notebook's EMPLOYED_SENTINEL / EMPLOYED_YEARS
  # columns exactly so downstream code stays consistent with the EDA.
  df[, EMPLOYED_SENTINEL := DAYS_EMPLOYED == 365243]
  df[, EMPLOYED_YEARS := fifelse(DAYS_EMPLOYED < 0, -DAYS_EMPLOYED / 365.25, NA_real_)]
  df
}

add_age_years <- function(df) {
  # EDA decision (notebook section "Age"): DAYS_BIRTH is a negative
  # day-offset from the application date. Converting to age in years makes
  # the feature interpretable to a non-technical reviewer and matches the
  # AGE_YEARS column already used for the age-band chart in the EDA.
  df[, AGE_YEARS := -DAYS_BIRTH / 365.25]
  df
}

add_financial_ratios <- function(df) {
  # EDA decision (notebook section "Loan amounts and income ratios"):
  # raw income and credit amounts overlap heavily across TARGET groups, but
  # income/credit ratios were kept as candidate features since the
  # credit-to-goods-price ratio showed the largest (still weak) correlation
  # with TARGET of the three ratios tested.
  df[, CREDIT_INCOME_RATIO  := AMT_CREDIT / AMT_INCOME_TOTAL]
  df[, ANNUITY_INCOME_RATIO := AMT_ANNUITY / AMT_INCOME_TOTAL]
  # AMT_GOODS_PRICE is missing for ~0.09% of applicants (cash loans with no
  # linked good); this ratio is intentionally NA for those rows rather than
  # a fabricated value.
  df[, CREDIT_GOODS_RATIO := AMT_CREDIT / AMT_GOODS_PRICE]
  df
}

add_external_source_count <- function(df) {
  # EDA decision (notebook sections "External source scores" and our own
  # EXT_SOURCE-completeness analysis): the NUMBER of external scores present
  # relates to TARGET in a non-monotonic way (applicants with 0 or all 3
  # scores present actually skew toward one pattern, applicants with 1 or 2
  # present skew toward another). This column captures data availability,
  # not applicant risk directly — kept as a feature, not a risk signal on
  # its own.
  df[, EXTERNAL_SOURCE_COUNT := rowSums(!is.na(.SD)),
     .SDcols = c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")]
  df
}

add_missing_indicators <- function(df, cols) {
  # EDA decision (notebook section "Missing values and the target"):
  # missingness rate itself differs by TARGET for several columns (property
  # fields showed roughly a 7-point gap). Rather than imputing silently and
  # discarding that signal, add one boolean flag per requested column.
  # NOTE: fread() reads a blank text field as "" rather than NA, so for text
  # columns (e.g., OCCUPATION_TYPE: 96,391 blanks, 0 NAs) a blank string must
  # count as missing too. Without this the flag is always FALSE.
  for (col in cols) {
    flag_name <- paste0(col, "_MISSING")
    x <- df[[col]]
    miss <- is.na(x)
    if (is.character(x)) miss <- miss | x == ""
    df[, (flag_name) := miss]
  }
  df
}

add_interactions <- function(df) {
  # Interaction terms motivated by the EDA's numeric-relationships ranking
  # (notebook section "Numeric relationships with the target"), where age
  # and the external scores were the strongest simple predictors.
  #
  # AGE_X_EMPLOYED: distinguishes a young applicant with long tenure from an
  # older applicant with the same tenure — plain age or tenure alone can't
  # tell these apart.
  df[, AGE_X_EMPLOYED := AGE_YEARS * EMPLOYED_YEARS]
  # DTI_X_EXT_COUNT: tests whether a high debt-to-income ratio matters more
  # or less when fewer external scores are available to corroborate it.
  df[, DTI_X_EXT_COUNT := ANNUITY_INCOME_RATIO * EXTERNAL_SOURCE_COUNT]
  df
}

# -----------------------------------------------------------------------------
# SECTION 2: Fit statistics — TRAINING DATA ONLY
#   This is the one function in the file that is allowed to look at
#   application_train and compute something from it. Nothing here touches
#   application_test, ever.
# -----------------------------------------------------------------------------

fit_preprocessing <- function(train) {
  # Returns a single list of all statistics needed to transform any dataset
  # (train or test) identically. Save this object (e.g. with saveRDS) so the
  # exact same values can be reused later without re-fitting.

  stats <- list()

  # --- Imputation values (medians), computed on train only ---
  # EDA decision: EXT_SOURCE_1/2/3, AMT_ANNUITY, AMT_GOODS_PRICE, and
  # CNT_FAM_MEMBERS all had non-trivial missingness in the EDA notebook.
  # Median imputation is used rather than mean because several of these
  # (credit amounts, family size) are right-skewed.
  impute_cols <- c("AMT_ANNUITY", "AMT_GOODS_PRICE",
                    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                    "CNT_FAM_MEMBERS", "DAYS_LAST_PHONE_CHANGE")
  stats$medians <- sapply(impute_cols, function(col) median(train[[col]], na.rm = TRUE))

  # --- Age band cut points ---
  # Matches the age bands used in the EDA notebook's age-vs-TARGET chart,
  # so engineered features stay traceable back to that analysis.
  stats$age_breaks <- c(20, 30, 40, 50, 60, 70)

  # --- Credit-income ratio quintile cut points, computed on train only ---
  # EDA decision: the EDA notebook split each EXT_SOURCE into quintiles to
  # compare default rate by score band. The same idea is applied here to
  # CREDIT_INCOME_RATIO to give models a discretized version of the ratio.
  train_ratio <- train$AMT_CREDIT / train$AMT_INCOME_TOTAL
  stats$credit_income_quintiles <- quantile(train_ratio, probs = seq(0, 1, 0.2), na.rm = TRUE)

  stats
}

# -----------------------------------------------------------------------------
# SECTION 3: Apply fitted statistics
#   These functions take the `stats` object from fit_preprocessing() and
#   apply it to any dataset. Calling these with the same `stats` object on
#   both train and test is what guarantees identical transformations.
# -----------------------------------------------------------------------------

apply_missing_imputation <- function(df, stats) {
  # Fills missing values using medians computed from TRAINING data only,
  # regardless of which dataset (train or test) is passed in here.
  for (col in names(stats$medians)) {
    if (col %in% names(df)) {
      val <- stats$medians[[col]]
      df[is.na(get(col)), (col) := val]
    }
  }
  df
}

add_age_band <- function(df, stats) {
  df[, AGE_BAND := cut(AGE_YEARS, breaks = stats$age_breaks,
                        right = FALSE, include.lowest = TRUE)]
  df
}

add_credit_income_quintile <- function(df, stats) {
  # Uses the quintile cut points fitted on train, so a given ratio value
  # falls into the same quintile bucket whether it appears in train or test.
  df[, CREDIT_INCOME_QUINTILE := cut(CREDIT_INCOME_RATIO,
                                      breaks = stats$credit_income_quintiles,
                                      include.lowest = TRUE, labels = FALSE)]
  df
}

# -----------------------------------------------------------------------------
# SECTION 4: Master pipeline
#   Single entry point that applies every step in the correct order. Always
#   copies the input so the caller's original data.table is never mutated
#   in place — this matters because train and test share this one function.
# -----------------------------------------------------------------------------

apply_preprocessing <- function(df, stats, is_train = FALSE) {
  df <- copy(df)

  df <- clean_days_employed(df)
  df <- add_age_years(df)
  df <- add_financial_ratios(df)
  df <- add_external_source_count(df)
  df <- add_missing_indicators(df, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                                      "AMT_ANNUITY", "AMT_GOODS_PRICE", "OCCUPATION_TYPE"))
  df <- apply_missing_imputation(df, stats)
  df <- add_age_band(df, stats)
  df <- add_credit_income_quintile(df, stats)
  df <- add_interactions(df)

  df
}

# -----------------------------------------------------------------------------
# SECTION 5: Train/test consistency check
#   Confirms the two prepared datasets have identical columns (other than
#   TARGET, which only exists in train) — the explicit requirement from the
#   assignment.
# -----------------------------------------------------------------------------

validate_train_test_consistency <- function(train_prepared, test_prepared) {
  train_cols <- setdiff(names(train_prepared), "TARGET")
  test_cols  <- names(test_prepared)

  missing_in_test <- setdiff(train_cols, test_cols)
  extra_in_test    <- setdiff(test_cols, train_cols)

  list(
    missing_in_test = missing_in_test,
    extra_in_test   = extra_in_test,
    identical       = length(missing_in_test) == 0 && length(extra_in_test) == 0
  )
}

# -----------------------------------------------------------------------------
# SECTION 6: Supplementary tables (OUT OF SCOPE — stub only)
#   The EDA notebook did not explore bureau.csv, previous_application.csv,
#   POS_CASH_balance.csv, credit_card_balance.csv, or installments_payments.csv
#   (explicitly optional at that stage). No aggregation or joining logic is
#   implemented here as a result. If that work is done later, the pattern
#   below is the safe way to add it: aggregate each supplementary table to
#   one row per SK_ID_CURR (a groupby, not a "fitted" statistic — no leakage
#   risk from this step itself), then left-join onto the application table.
# -----------------------------------------------------------------------------

# aggregate_bureau <- function(bureau) {
#   # Example shape only — not implemented, since bureau.csv was not part of
#   # this project's EDA scope.
#   bureau[, .(
#     BUREAU_LOAN_COUNT     = .N,
#     BUREAU_ACTIVE_COUNT   = sum(CREDIT_ACTIVE == "Active"),
#     BUREAU_AVG_DAYS_CREDIT = mean(DAYS_CREDIT, na.rm = TRUE)
#   ), by = SK_ID_CURR]
# }
#
# join_bureau_features <- function(df, bureau_agg) {
#   merge(df, bureau_agg, by = "SK_ID_CURR", all.x = TRUE)
# }

# =============================================================================
# SECTION 7: Run the pipeline on the Home Credit files
#   Running this whole file (for example with Source in Positron) defines the
#   functions above and then runs them on the data. Run it from the repo root,
#   with application_train.csv and application_test.csv in data/raw/.
#
#   Other scripts that only need the functions (such as the modeling notebook)
#   set SKIP_PIPELINE <- TRUE before calling source() to skip this section.
# =============================================================================

if (!exists("SKIP_PIPELINE") || !isTRUE(SKIP_PIPELINE)) {

  train_path <- "data/raw/application_train.csv"
  test_path  <- "data/raw/application_test.csv"
  if (!file.exists(train_path) || !file.exists(test_path)) {
    stop("Could not find the data files. Run this from the repo root and put ",
         "application_train.csv and application_test.csv in data/raw/. ",
         "Current folder: ", getwd())
  }

  train <- fread(train_path)
  test  <- fread(test_path)

  # Fit on training data only, then apply the same statistics to train and test
  prep_stats  <- fit_preprocessing(train)
  train_ready <- apply_preprocessing(train, prep_stats, is_train = TRUE)
  test_ready  <- apply_preprocessing(test,  prep_stats, is_train = FALSE)

  # Train and test must have identical columns (except TARGET)
  check <- validate_train_test_consistency(train_ready, test_ready)
  stopifnot(check$identical)

  # Save outputs (these folders are git-ignored; do not commit them)
  dir.create("models", showWarnings = FALSE)
  saveRDS(prep_stats, "models/preprocessing_stats.rds")
  dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
  fwrite(train_ready, "data/processed/application_train_prepared.csv")
  fwrite(test_ready,  "data/processed/application_test_prepared.csv")

  # Summary of what the pipeline did
  cat("Train:", nrow(train_ready), "rows,", ncol(train_ready), "columns\n")
  cat("Test: ", nrow(test_ready),  "rows,", ncol(test_ready),  "columns\n")
  cat("Train/test columns consistent:", check$identical, "\n")
  cat("New columns added:", paste(setdiff(names(train_ready), names(train)), collapse = ", "), "\n")
  cat("OCCUPATION_TYPE_MISSING counts:\n")
  print(table(train_ready$OCCUPATION_TYPE_MISSING))
}
