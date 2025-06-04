import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
output_dir = Path('Output/7.regional_regression')
output_dir.mkdir(parents=True, exist_ok=True)

def load_cluster_assignments():
    """Load cluster assignments from code 6 output"""
    with open('Output/6.clustering/6.cluster_characteristics.json', 'r') as f:
        clusters = json.load(f)
    return clusters

def stepwise_selection_with_logging(X, y, relevant_lags_dict, positive_vars, negative_vars, region_name, sign_tolerance=0.05, 
                                  initial_list=[], threshold_in=0.05, threshold_out=0.05, criterion='aic'):
    """
    Perform stepwise selection with logging for regional analysis
    """
    # Always create or clear debug_output.txt at the start
    with open('debug_output.txt', 'w') as dbg:
        dbg.write(f"Debug log for region: {region_name}\n")
    
    included = list(initial_list)
    dropped_vars = set()
    best_criterion = None
    best_model_vars = list(included)
    excluded_vars = set()
    
    def log_step(msg):
        logger.info(msg)
        with open(output_dir / f'variable_selection_{region_name}.log', 'a') as f:
            f.write(msg + '\n')
    
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included) - dropped_vars - excluded_vars)
        candidates = []
        for new_column in excluded:
            var_base = new_column.split('_')[0]
            if var_base in relevant_lags_dict:
                lags_to_try = relevant_lags_dict[var_base]
                found_valid = False
                lag_scores = []
                for lag in lags_to_try:
                    if lag == 0:
                        colname = var_base
                    elif lag > 0:
                        colname = f"{var_base}_lag{lag}"
                    else:
                        colname = f"{var_base}_lead{abs(lag)}"
                    if colname not in X.columns or colname in included or colname in dropped_vars or colname in excluded_vars:
                        continue
                    # Prepare data for regression
                    X_reg = X[included + [colname]].apply(pd.to_numeric, errors='coerce')
                    y_reg = y.apply(pd.to_numeric, errors='coerce')
                    valid_idx = X_reg.notna().all(axis=1) & y_reg.notna()
                    if valid_idx.sum() < 10:
                        continue
                    X_reg = X_reg[valid_idx]
                    y_reg = y_reg[valid_idx]
                    # Convert y_reg (Series) to DataFrame so that select_dtypes can be called
                    y_reg_df = y_reg.to_frame()
                    # Always write debug info before regression
                    with open('debug_output.txt', 'a') as dbg:
                        dbg.write(f"DEBUG: X_reg dtypes: {X_reg.dtypes}\n")
                        dbg.write(f"DEBUG: y_reg dtypes: {y_reg_df.dtypes}\n")
                        dbg.write(f"DEBUG: X_reg shape: {X_reg.shape}, y_reg shape: {y_reg_df.shape}\n")
                        dbg.write(f"DEBUG: X_reg index: {X_reg.index}, y_reg index: {y_reg_df.index}\n")
                        object_cols_X = X_reg.select_dtypes(include=['object']).columns.tolist()
                        object_cols_y = y_reg_df.select_dtypes(include=['object']).columns.tolist()
                        if object_cols_X:
                            dbg.write(f"DEBUG: X_reg object columns: {object_cols_X}\n")
                        if object_cols_y:
                            dbg.write(f"DEBUG: y_reg object columns: {object_cols_y}\n")
                    X_reg = X_reg.astype(float)
                    model = sm.OLS(y_reg, sm.add_constant(X_reg)).fit()
                    coef = model.params.get(colname, np.nan)
                    sign_ok = True
                    if var_base in positive_vars and coef < -sign_tolerance * abs(coef):
                        sign_ok = False
                    if var_base in negative_vars and coef > sign_tolerance * abs(coef):
                        sign_ok = False
                    if criterion == 'aic':
                        score = model.aic
                    else:
                        score = model.bic
                    lag_scores.append((colname, score, sign_ok, coef))
                    if sign_ok and not found_valid:
                        candidates.append((score, colname))
                        found_valid = True
                if not found_valid:
                    log_step(f"Drop {var_base}: all relevant lags/leads violate sign constraint. Tried: " + 
                            ", ".join([f'{name} (score={score:.3f}, sign_ok={sign_ok}, coef={coef:.4f})' 
                                     for name, score, sign_ok, coef in lag_scores]))
                    for lag in lags_to_try:
                        if lag == 0:
                            colname = var_base
                        elif lag > 0:
                            colname = f"{var_base}_lag{lag}"
                        else:
                            colname = f"{var_base}_lead{abs(lag)}"
                        excluded_vars.add(colname)
                else:
                    log_step(f"Candidates for {var_base}: " + 
                            ", ".join([f'{name} (score={score:.3f}, sign_ok={sign_ok}, coef={coef:.4f})' 
                                     for name, score, sign_ok, coef in lag_scores]))
            else:
                # Prepare data for regression
                X_reg = X[included + [new_column]].apply(pd.to_numeric, errors='coerce')
                y_reg = y.apply(pd.to_numeric, errors='coerce')
                valid_idx = X_reg.notna().all(axis=1) & y_reg.notna()
                if valid_idx.sum() < 10:
                    continue
                X_reg = X_reg[valid_idx]
                y_reg = y_reg[valid_idx]
                # Convert y_reg (Series) to DataFrame so that select_dtypes can be called
                y_reg_df = y_reg.to_frame()
                # Always write debug info before regression
                with open('debug_output.txt', 'a') as dbg:
                    dbg.write(f"DEBUG: X_reg dtypes (else branch): {X_reg.dtypes}\n")
                    dbg.write(f"DEBUG: y_reg dtypes (else branch): {y_reg_df.dtypes}\n")
                    dbg.write(f"DEBUG: X_reg shape (else branch): {X_reg.shape}, y_reg shape (else branch): {y_reg_df.shape}\n")
                    dbg.write(f"DEBUG: X_reg index (else branch): {X_reg.index}, y_reg index (else branch): {y_reg_df.index}\n")
                    object_cols_X = X_reg.select_dtypes(include=['object']).columns.tolist()
                    object_cols_y = y_reg_df.select_dtypes(include=['object']).columns.tolist()
                    if object_cols_X:
                        dbg.write(f"DEBUG: X_reg object columns (else branch): {object_cols_X}\n")
                    if object_cols_y:
                        dbg.write(f"DEBUG: y_reg object columns (else branch): {object_cols_y}\n")
                X_reg = X_reg.astype(float)
                model = sm.OLS(y_reg, sm.add_constant(X_reg)).fit()
                if criterion == 'aic':
                    score = model.aic
                else:
                    score = model.bic
                candidates.append((score, new_column))
        
        if candidates:
            log_step("All candidates this step: " + 
                    ", ".join([f'{c[1]} (score={c[0]:.3f})' for c in candidates]))
            candidates.sort()
            best_score, best_feature = candidates[0]
            if best_criterion is None or best_score < best_criterion:
                included.append(best_feature)
                best_criterion = best_score
                best_model_vars = list(included)
                changed = True
                log_step(f'Add  {best_feature:15} with {criterion.upper()} {best_score:.3f}')
        
        # Backward step
        if len(included) > 0:
            candidates = []
            for col in included:
                vars_minus = [v for v in included if v != col]
                if not vars_minus:
                    continue
                # Prepare data for regression
                X_reg = X[vars_minus].apply(pd.to_numeric, errors='coerce')
                y_reg = y.apply(pd.to_numeric, errors='coerce')
                valid_idx = X_reg.notna().all(axis=1) & y_reg.notna()
                if valid_idx.sum() < 10:
                    continue
                X_reg = X_reg[valid_idx]
                y_reg = y_reg[valid_idx]
                # Convert y_reg (Series) to DataFrame so that select_dtypes can be called
                y_reg_df = y_reg.to_frame()
                # Always write debug info before regression
                with open('debug_output.txt', 'a') as dbg:
                    dbg.write(f"DEBUG: X_reg dtypes (backward step): {X_reg.dtypes}\n")
                    dbg.write(f"DEBUG: y_reg dtypes (backward step): {y_reg_df.dtypes}\n")
                    dbg.write(f"DEBUG: X_reg shape (backward step): {X_reg.shape}, y_reg shape (backward step): {y_reg_df.shape}\n")
                    dbg.write(f"DEBUG: X_reg index (backward step): {X_reg.index}, y_reg index (backward step): {y_reg_df.index}\n")
                    object_cols_X = X_reg.select_dtypes(include=['object']).columns.tolist()
                    object_cols_y = y_reg_df.select_dtypes(include=['object']).columns.tolist()
                    if object_cols_X:
                        dbg.write(f"DEBUG: X_reg object columns (backward step): {object_cols_X}\n")
                    if object_cols_y:
                        dbg.write(f"DEBUG: y_reg object columns (backward step): {object_cols_y}\n")
                X_reg = X_reg.astype(float)
                model = sm.OLS(y_reg, sm.add_constant(X_reg)).fit()
                if criterion == 'aic':
                    score = model.aic
                else:
                    score = model.bic
                candidates.append((score, col))
            if candidates:
                candidates.sort()
                best_score, worst_feature = candidates[0]
                if best_score < best_criterion:
                    included.remove(worst_feature)
                    dropped_vars.add(worst_feature)
                    best_criterion = best_score
                    best_model_vars = list(included)
                    changed = True
                    log_step(f'Drop {worst_feature:15} with {criterion.upper()} {best_score:.3f}')
        
        if not changed:
            break
    
    return best_model_vars

def analyze_region(data, region_name, countries, macro_vars, positive_vars=None, negative_vars=None, 
                  sign_tolerance=0.05, relevant_lags_dict=None):
    """
    Analyze a region by stacking countries and performing regression
    """
    logger.info(f"\nAnalyzing region: {region_name}")
    logger.info(f"Countries: {', '.join(countries)}")
    
    # Filter data for countries in this region
    region_data = data[data['cinc'].isin(countries)].copy()
    
    if len(region_data) == 0:
        logger.warning(f"No data found for region {region_name}")
        return None
    
    # Prepare variables
    X = region_data[macro_vars].copy()
    y = region_data[['dlnPD']].copy()
    
    # Add country fixed effects
    X = pd.get_dummies(X, columns=['cinc'], drop_first=True)
    # Ensure all columns in X are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Drop rows with NA in X or y
    valid_idx = X.notna().all(axis=1) & y['dlnPD'].notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Perform stepwise selection with logging
    with open(output_dir / f'variable_selection_{region_name}.log', 'w') as f:
        f.write(f"Variable Selection Process for {region_name}\n")
        f.write("=" * 50 + "\n\n")
    
    selected_vars = stepwise_selection_with_logging(X, y['dlnPD'], relevant_lags_dict, 
                                                  positive_vars, negative_vars, region_name)
    
    if not selected_vars:
        logger.warning(f"No significant variables found for region {region_name}")
        return None
    
    # Fit final model with selected variables
    X_selected = X[selected_vars]
    model = sm.OLS(y['dlnPD'], sm.add_constant(X_selected)).fit()
    
    # Diagnostic tests
    resid = model.resid
    het_test = het_breuschpagan(resid, model.model.exog)
    dw_stat = durbin_watson(resid)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Raw vs Fitted
    axes[0,0].scatter(y['dlnPD'], model.fittedvalues, alpha=0.5)
    axes[0,0].plot([y['dlnPD'].min(), y['dlnPD'].max()], 
                   [y['dlnPD'].min(), y['dlnPD'].max()], 'r--')
    axes[0,0].set_xlabel('Raw dlnPD')
    axes[0,0].set_ylabel('Fitted dlnPD')
    axes[0,0].set_title('Raw vs Fitted Values')
    
    # Residuals vs Fitted
    axes[0,1].scatter(model.fittedvalues, resid)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Fitted values')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residuals vs Fitted')
    
    # QQ Plot
    stats.probplot(resid, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Normal Q-Q')
    
    # Residuals vs Time
    axes[1,1].plot(resid.index, resid)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Time')
    
    # Save results
    plt.tight_layout()
    plt.savefig(output_dir / f'diagnostics_{region_name}.png')
    plt.close()
    
    # Save model summary
    with open(output_dir / f'model_summary_{region_name}.txt', 'w') as f:
        f.write(f"Model Summary for {region_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\nDiagnostic Tests:\n")
        f.write(f"Breusch-Pagan Heteroskedasticity Test: {het_test}\n")
        f.write(f"Durbin-Watson Statistic: {dw_stat}\n")
    
    return model

def main():
    # Load data
    data = pd.read_csv('Output/4.transformation/transformed_data.csv')
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    # Add dlnPD_lag and lnPD_TTC_gap as in code 5
    data['dlnPD_lag'] = data.groupby('cinc')['dlnPD'].shift(1)
    data['lnPD_TTC_gap'] = data.groupby('cinc')['lnPD'].transform(lambda x: x - x.mean())
    
    # Load cluster assignments
    clusters = load_cluster_assignments()
    
    # Define variables and parameters
    macro_vars = ['FGDPL$Q_trans', 'FCPIQ_trans', 'FGGDEBTGDPQ_trans', 'FNETEXGSD$Q_trans',
                 'FTFXIUSAQ_trans', 'FLBRQ_trans', 'FSTOCKPQ_trans', 'FCPWTI.IUSA_trans',
                 'FRGT10YQ_trans', 'lnPD_TTC_gap', 'dlnPD_lag', 'cinc']
    
    positive_vars = ['FGDPL$Q', 'FNETEXGSD$Q', 'FTFXIUSAQ', 'FSTOCKPQ']
    negative_vars = ['FCPIQ', 'FGGDEBTGDPQ', 'FLBRQ', 'FCPWTI.IUSA', 'FRGT10YQ']
    
    relevant_lags_dict = { mv: [-4, -3, -2, -1, 0, 1, 2, 3, 4] for mv in [ 'FGDPL$Q', 'FCPIQ', 'FGGDEBTGDPQ', 'FNETEXGSD$Q', 'FTFXIUSAQ', 'FLBRQ', 'FSTOCKPQ', 'FCPWTI.IUSA', 'FRGT10YQ' ] }
    
    # Analyze each region
    for region_name, region_info in clusters.items():
        logger.info(f"\nProcessing region: {region_name}")
        model = analyze_region(data, region_name, region_info['countries'], 
                             macro_vars, positive_vars, negative_vars,
                             relevant_lags_dict=relevant_lags_dict)
        
        if model is not None:
            logger.info(f"Completed analysis for {region_name}")
        else:
            logger.warning(f"Analysis failed for {region_name}")

if __name__ == "__main__":
    main() 