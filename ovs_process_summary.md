# OVS (Optimal Variable Selection) Process — Detailed Documentation

## Overview

The OVS (Optimal Variable Selection) process is a systematic, automated approach for selecting the best regression models from a large set of possible variable combinations. It is designed to ensure robust, interpretable, and statistically sound models, especially in contexts where domain knowledge imposes additional constraints (e.g., expected coefficient signs, variable usage, or multicollinearity).

The process is implemented in R (see `ovs.R`) and has been replicated in Python for integration with modern data science workflows.

---

## Key Features and Parameters

| Parameter         | Description                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------|
| `depvar`          | The dependent variable (target) for regression.                                              |
| `drivers`         | List of candidate predictor variables.                                                       |
| `rhs_fixed`       | Variables that must be included in every model (subject to constraints).                     |
| `no_track`        | Variables that must be included, regardless of statistical constraints (e.g., p-value).      |
| `negatives`       | Variables expected to have negative coefficients.                                            |
| `positives`       | Variables expected to have positive coefficients.                                            |
| `k`               | Maximum number of variables (excluding fixed terms) in a model.                              |
| `cc`              | Correlation threshold (e.g., 0.75) for removing highly correlated variable pairs.            |
| `pval`            | Maximum allowed p-value for variables (default 0.1).                                         |
| `varSingleUse`    | Ensures only one transformation/lag of a variable is used per model.                         |
| `noconst`         | If `TRUE`, fit models without an intercept.                                                  |
| `family`          | Regression family (e.g., OLS, GLM).                                                         |
| `write_file`      | Optional: path to save results as CSV/RDS.                                                   |

---

## Step-by-Step Process

### 1. **Input Validation and Preprocessing**
- Checks that all specified variables exist in the data.
- Removes rows with missing values in any required variable (including fixed and no-track variables).
- Optionally, filters data by time or other criteria.

### 2. **Variable Permutation Generation**
- Generates all possible combinations of up to `k` variables from the candidate set.
- For each combination, appends any `rhs_fixed` and `no_track` variables (ensuring no duplicates).
- Each combination represents a candidate model.

### 3. **Constraint Filtering**

#### a. **Single-Use Constraint (`varSingleUse`)**
- Ensures that only one transformation/lag of a variable is used in any model.
- Removes models where the same base variable appears more than once (e.g., both GDP_lag1 and GDP_lag2).

#### b. **Correlation Constraint (`cc`)**
- Computes the correlation matrix for all variables in each model.
- Removes models where any pair of variables exceeds the specified correlation threshold.
- Special handling for `rhs_fixed` and `no_track` variables (can be exempted from this check).

#### c. **Regression and P-value Filtering**
- Fits the specified regression (OLS or GLM) for each remaining model.
- Removes models where any variable (except those in `no_track`) has a p-value above the threshold (`pval`).
- Optionally, can enforce sign constraints (see below).

#### d. **Sign Constraints (`negatives`, `positives`)**
- For variables listed in `negatives`, ensures their estimated coefficients are negative.
- For variables listed in `positives`, ensures their estimated coefficients are positive.
- Removes models violating these sign expectations.

### 4. **Model Evaluation and Output**
For each surviving model:
- **Model Specification:** List of variables (including lags/transformations).
- **Coefficients:** Estimated values for each variable.
- **P-values:** Statistical significance for each coefficient.
- **Model Statistics:**  
  - R-squared, Adjusted R-squared  
  - AIC, BIC (for model comparison)  
  - Error metrics: MAE, MSE, RMSE, MAPE, etc.
- **Diagnostics:**  
  - Residual analysis  
  - Multicollinearity (VIF)  
  - Plots: coefficient bar plots, fitted vs. actual, residuals, etc.

- **Output Files:**  
  - Detailed results for all models (CSV/JSON)  
  - Summary of top models per country (CSV)  
  - Diagnostic plots (PNG)  
  - Optionally, RDS files for R compatibility

---

## Example Workflow

1. **Prepare Data:**  
   - Ensure all macro variables are transformed and aligned (e.g., only Baseline scenario columns).
   - Compute lagged dependent variable and mean-reverting term for each country.

2. **Configure OVS Parameters:**  
   - Set `k` (max variables), `cc` (correlation threshold), `pval` (significance), etc.
   - Specify any sign constraints or fixed variables.

3. **Run OVS Process:**  
   - For each country, generate all valid model combinations.
   - Apply all constraints and fit regressions.
   - Save and review results.

4. **Interpret Results:**  
   - Examine top models by adjusted R-squared or other criteria.
   - Review coefficient signs, p-values, and diagnostics.
   - Use plots to assess model fit and residuals.

---

## Output Structure

- **Model Table:**  
  | Model | Variables | Coefficients | P-values | R² | Adj. R² | AIC | BIC | MAE | RMSE | ... |
  |-------|-----------|--------------|----------|----|---------|-----|-----|-----|------|-----|

- **Diagnostics:**  
  - Coefficient bar plots for top models  
  - Fitted vs. actual plots  
  - Residual plots

- **Files:**  
  - `ovs_results_{country}.csv` — All models for a country  
  - `ovs_all_results.csv` — All models, all countries  
  - `coef_plot_{country}.png`, `fitted_vs_actual_{country}.png` — Plots

---

## Best Practices and Notes

- **Interpretability:**  
  OVS is designed to produce models that are both statistically sound and interpretable, with explicit control over variable selection and constraints.

- **Extensibility:**  
  The process can be extended to include additional constraints (e.g., group lasso, custom error metrics) or to support other model types.

- **Performance:**  
  For large numbers of variables or lags, the number of model combinations can be very large. Consider parallelization or pre-filtering variables.

- **Reproducibility:**  
  All steps, parameters, and outputs are logged for reproducibility and auditability.

---

## References

- See the R implementation in `ovs.R` for the original logic and additional comments.
- Python replication is available in `Code/6.ovs_variable_selection.py`.

--- 