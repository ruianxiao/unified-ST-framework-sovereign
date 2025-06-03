import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def calculate_qdiff(df, columns):
    """
    Calculate quarterly differences for specified columns
    """
    return df[columns].diff()

def check_stationarity(series):
    """
    Perform ADF test and return results
    """
    result = adfuller(series.dropna())
    return {
        'adf_stat': float(result[0]),
        'p_value': float(result[1]),
        'is_stationary': bool(result[1] < 0.05)
    }

def calculate_vif(X):
    """
    Calculate VIF for each variable
    """
    if X.shape[1] <= 1:
        # If only one variable, VIF is undefined (return 1.0 by convention)
        return pd.DataFrame({
            "Variable": X.columns,
            "VIF": [1.0] * X.shape[1]
        })
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [float(variance_inflation_factor(X.values, i)) for i in range(X.shape[1])]
    return vif_data

def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05):
    """
    Perform stepwise selection to identify significant variables
    """
    included = list(initial_list)
    while True:
        changed = False
        
        # Forward step
        excluded = list(set(X.columns) - set(included))
        if excluded:
            new_pval = pd.Series(index=excluded, dtype=float)
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
                new_pval[new_column] = model.pvalues.iloc[-1]  # Get p-value for the last variable (new one)
            best_pval = new_pval.min()
            
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                print(f'Add  {best_feature:15} with p-value {best_pval:.6}')
        
        # Backward step
        if len(included) > 0:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            # remove constant term's p-value
            if len(included) > 1:
                pvalues = model.pvalues.iloc[1:]  # Skip constant
            else:
                pvalues = pd.Series(model.pvalues.iloc[-1], index=[included[0]])
            
            worst_pval = pvalues.max()
            
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                print(f'Drop {worst_feature:15} with p-value {worst_pval:.6}')
            
        if not changed:
            break
            
    return included

def analyze_country(data, country_code, macro_vars, output_dir, positive_vars=None, negative_vars=None, sign_tolerance=0.05, relevant_lags_dict=None):
    """
    Perform regression analysis for a single country, with sign constraints
    """
    # Create a log file for this country
    log_file = output_dir / f'variable_selection_{country_code}.log'
    with open(log_file, 'w') as f:
        f.write(f"Variable Selection Process for {country_code}\n")
        f.write("=" * 50 + "\n\n")
        
        def log_step(message):
            print(message)
            f.write(message + "\n")
            f.flush()
            
        log_step(f"\nAnalyzing {country_code}")
        if positive_vars is None:
            positive_vars = []
        if negative_vars is None:
            negative_vars = []
        try:
            # Filter data for the country
            country_data = data[data['cinc'] == country_code].sort_values('yyyyqq')
            if len(country_data) < 20:
                print(f"Insufficient data points for {country_code}")
                return None
            # Use already transformed variables
            X = country_data[macro_vars]
            y = country_data[['dlnPD']]  # Use delta log PD as dependent variable
            # Drop NA values
            valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]
            if len(X) < 20:
                print(f"Insufficient data points for {country_code}")
                return None
            # Check stationarity
            stationarity_results = {}
            for col in macro_vars:
                stationarity_results[col] = check_stationarity(X[col])
            stationarity_results['dlnPD'] = check_stationarity(y['dlnPD'])
            # Calculate VIF before selection
            vif_before = calculate_vif(X)
            # Perform stepwise selection with logging
            def stepwise_selection_with_logging(X, y, relevant_lags_dict, positive_vars, negative_vars, sign_tolerance=0.05, initial_list=[], threshold_in=0.05, threshold_out=0.05, criterion='aic'):
                included = list(initial_list)
                dropped_vars = set()
                best_criterion = None
                best_model_vars = list(included)
                excluded_vars = set()
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
                                model = sm.OLS(y, sm.add_constant(X[included + [colname]])).fit()
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
                                    # break  # Only add the first valid lag/lead (keep collecting scores for logging)
                            # Enhanced logging for sign constraint violations
                            if not found_valid:
                                log_step(f"Drop {var_base}: all relevant lags/leads violate sign constraint. Tried: " + ", ".join([f'{name} (score={score:.3f}, sign_ok={sign_ok}, coef={coef:.4f})' for name, score, sign_ok, coef in lag_scores]))
                                for lag in lags_to_try:
                                    if lag == 0:
                                        colname = var_base
                                    elif lag > 0:
                                        colname = f"{var_base}_lag{lag}"
                                    else:
                                        colname = f"{var_base}_lead{abs(lag)}"
                                    excluded_vars.add(colname)
                            else:
                                log_step(f"Candidates for {var_base}: " + ", ".join([f'{name} (score={score:.3f}, sign_ok={sign_ok}, coef={coef:.4f})' for name, score, sign_ok, coef in lag_scores]))
                        else:
                            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
                            if criterion == 'aic':
                                score = model.aic
                            else:
                                score = model.bic
                            candidates.append((score, new_column))
                    if candidates:
                        # Enhanced logging: log all candidates and their scores
                        log_step("All candidates this step: " + ", ".join([f'{c[1]} (score={c[0]:.3f})' for c in candidates]))
                        candidates.sort()
                        best_score, best_feature = candidates[0]
                        if best_criterion is None or best_score < best_criterion:
                            included.append(best_feature)
                            best_criterion = best_score
                            best_model_vars = list(included)
                            changed = True
                            log_step(f'Add  {best_feature:15} with {criterion.upper()} {best_score:.3f}')
                    # Backward step (no sign constraint check needed)
                    if len(included) > 0:
                        candidates = []
                        for col in included:
                            vars_minus = [v for v in included if v != col]
                            if not vars_minus:
                                continue
                            model = sm.OLS(y, sm.add_constant(X[vars_minus])).fit()
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

            # Use the logging version of stepwise selection
            selected_vars = stepwise_selection_with_logging(X, y['dlnPD'], relevant_lags_dict, positive_vars, negative_vars)
            if not selected_vars:
                print(f"No significant variables found for {country_code}")
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
            axes[0,0].plot([y['dlnPD'].min(), y['dlnPD'].max()], [y['dlnPD'].min(), y['dlnPD'].max()], 'r--')
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
            # Histogram
            axes[2,0].hist(resid, bins=30, density=True, alpha=0.7)
            xmin, xmax = axes[2,0].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, np.mean(resid), np.std(resid))
            axes[2,0].plot(x, p, 'k', linewidth=2)
            axes[2,0].set_title('Residuals Distribution')
            # Time series of raw and fitted
            axes[2,1].plot(y.index, y['dlnPD'], 'b-', alpha=0.5, label='Raw')
            axes[2,1].plot(y.index, model.fittedvalues, 'r-', alpha=0.5, label='Fitted')
            axes[2,1].set_xlabel('Time')
            axes[2,1].set_ylabel('dlnPD')
            axes[2,1].set_title('Time Series: Raw vs Fitted')
            axes[2,1].legend()
            plt.tight_layout()
            plt.savefig(output_dir / f'diagnostics_{country_code}.png')
            plt.close()
            # Convert all numpy types to Python native types for JSON serialization
            results = {
                'country': country_code,
                'n_obs': int(len(X_selected)),
                'r2': float(model.rsquared),
                'adj_r2': float(model.rsquared_adj),
                'selected_vars': selected_vars,
                'coefficients': {k: float(v) for k, v in model.params.to_dict().items()},
                'p_values': {k: float(v) for k, v in model.pvalues.to_dict().items()},
                'het_test_p': float(het_test[1]),
                'durbin_watson': float(dw_stat),
                'vif_before': vif_before.to_dict('records'),
                'vif_after': calculate_vif(X_selected).to_dict('records'),
                'stationarity': stationarity_results,
            }
            return results
        except Exception as e:
            print(f"Error analyzing {country_code}: {str(e)}")
            return None

def check_constant_ratio(series):
    """
    Check if a series has the same value for more than 70% of the time.
    Returns True if the most common value appears more than 70% of the time.
    """
    if series.empty:
        return True
    value_counts = series.value_counts()
    if value_counts.empty:
        return True
    most_common_ratio = value_counts.iloc[0] / len(series)
    return most_common_ratio > 0.7

def univariate_regression_country(data, country_code, macro_vars, output_dir, positive_vars=None, negative_vars=None, sign_tolerance=0.05, max_lag=4, max_lead=4):
    """
    For each macro variable, run univariate regression of dlnPD on that macro variable (with up to 4 leads/lags),
    apply sign restriction, and decide if the macro variable is relevant for this country.
    Save results as a dict.
    """
    results = {}
    dropped_vars = []  # Track dropped variables
    country_data = data[data['cinc'] == country_code].sort_values('yyyyqq').copy()
    if len(country_data) < 20:
        return None, []
    y = country_data['dlnPD']
    for var in macro_vars:
        # Drop variable if more than 70% of values are the same
        var_series = country_data[var] if var in country_data.columns else None
        if var_series is not None:
            if check_constant_ratio(var_series):
                dropped_vars.append(var)
                continue  # Skip this variable
        var_results = []
        for lag in range(-max_lead, max_lag+1):
            if lag == 0:
                var_name = var
                x = country_data[var]
            elif lag > 0:
                var_name = f"{var}_lag{lag}"
                x = country_data[var].shift(lag)
            else:
                var_name = f"{var}_lead{abs(lag)}"
                x = country_data[var].shift(lag)
            # Align and drop NA
            valid = x.notna() & y.notna()
            if valid.sum() < 20:
                continue
            X = sm.add_constant(x[valid])
            y_valid = y[valid]
            try:
                model = sm.OLS(y_valid, X).fit()
                coef = model.params[var]
                pval = model.pvalues[var]
                sign_ok = True
                var_base = var.split('_')[0]
                # Only apply sign constraint if in positive_vars or negative_vars
                if positive_vars and var_base in positive_vars and coef < -sign_tolerance * abs(coef):
                    sign_ok = False
                if negative_vars and var_base in negative_vars and coef > sign_tolerance * abs(coef):
                    sign_ok = False
                # If not in either, do not apply sign constraint (sign_ok remains True)
                relevant = (pval < 0.05) and sign_ok
                # Ensure all values are Python native types
                var_results.append({
                    'var': str(var),
                    'lag': int(lag),
                    'coef': float(coef),
                    'pval': float(pval),
                    'r2': float(model.rsquared),
                    'sign_ok': bool(sign_ok),
                    'relevant': bool(relevant)
                })
            except Exception as e:
                continue
        results[var] = var_results
    return results, dropped_vars

def main():
    # Create output directory
    output_dir = Path('Output/5.regression_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load transformed data
    print("Loading regression data...")
    data = pd.read_csv('Output/4.transformation/transformed_data.csv')
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    # Add lagged dependent variable and mean-reverting term
    data['dlnPD_lag'] = data.groupby('cinc')['dlnPD'].shift(1)
    # Redefine lnPD_TTC_gap as the difference between current log PD and long-term average log PD for each country
    data['lnPD_TTC_gap'] = data.groupby('cinc')['lnPD'].transform(lambda x: x - x.mean())
    # List of macro variables (using transformed variables, exclude lag/mean-reverting terms for univariate)
    macro_vars = ['FCPIQ_trans', 'FGD$Q_trans', 'FGDPL$Q_trans', 'FGGDEBTGDPQ_trans', 'FLBRQ_trans',
                 'FNETEXGSD$Q_trans', 'FRGT10YQ_trans', 'FSTOCKPQ_trans', 'FTFXIUSAQ_trans',
                 'FCPWTI.IUSA_trans']
    # Define sign constraints
    positive_vars = [
        'FLBRQ',      # Unemployment Rate
        'FCPIQ',      # Inflation Rate
        'FRGT10YQ',   # Interest Rate (10Y Government Bond)
        'FTFXIUSAQ',  # Exchange Rate
        'FGGDEBTGDPQ' # Debt to GDP Ratio
    ]
    negative_vars = [
        'FGDPL$Q',    # GDP Growth
        'FNETEXGSD$Q' # Net Exports
    ]
    # Step 5.1: Univariate regression
    print("Running step 5.1: Univariate regression...")
    univariate_results = {}
    dropped_summary = {}
    for country in data['cinc'].unique():
        res, dropped_vars = univariate_regression_country(data, country, macro_vars, output_dir, positive_vars, negative_vars)
        if res:
            univariate_results[country] = res
        if dropped_vars:
            dropped_summary[country] = dropped_vars
    with open(output_dir / 'univariate_results.json', 'w') as f:
        json.dump(univariate_results, f, indent=2)
    # Save dropped variable summary
    with open(output_dir / 'dropped_variables_summary.json', 'w') as f:
        json.dump(dropped_summary, f, indent=2)
    # Save a summary
    with open(output_dir / 'univariate_summary.txt', 'w') as f:
        for country, res in univariate_results.items():
            f.write(f"Country: {country}\n")
            for var, var_res in res.items():
                relevant_lags = [r for r in var_res if r['relevant']]
                if relevant_lags:
                    f.write(f"  {var}: relevant at lags {[r['lag'] for r in relevant_lags]}\n")
            f.write("\n")
    # Plot time series for dropped variables
    for country, dropped_vars in dropped_summary.items():
        country_data = data[data['cinc'] == country].sort_values('yyyyqq')
        for var in dropped_vars:
            if var in country_data.columns:
                plt.figure(figsize=(10, 4))
                plt.plot(country_data['yyyyqq'], country_data[var], marker='o', linestyle='-', label=var)
                plt.title(f"{var} time series for {country}")
                plt.xlabel('yyyyqq')
                plt.ylabel(var)
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f'dropped_{country}_{var}.png')
                plt.close()

    # Step 5.2: Prepare lag/lead columns for all relevant macro variables
    print("Preparing lag/lead columns for multivariate regression...")
    # Collect all relevant (var, lag) pairs across all countries
    relevant_var_lags = set()
    for country, res in univariate_results.items():
        for var, lags in res.items():
            for r in lags:
                if r['relevant']:
                    relevant_var_lags.add((var, r['lag']))
    # Create these columns in the DataFrame if not already present
    for var, lag in relevant_var_lags:
        if lag == 0:
            continue  # already present
        colname = f"{var}_lag{lag}" if lag > 0 else f"{var}_lead{abs(lag)}"
        if colname not in data.columns:
            if lag > 0:
                data[colname] = data.groupby('cinc')[var].shift(lag)
            else:
                data[colname] = data.groupby('cinc')[var].shift(lag)

    # Load univariate results for relevant lags
    with open(output_dir / 'univariate_results.json', 'r') as f:
        univariate_results = json.load(f)
    # Step 5.2: Multivariate regression (existing logic)
    print("Running step 5.2: Multivariate regression...")
    all_results = {}
    for country in data['cinc'].unique():
        # Build relevant lags dict for this country
        relevant_lags_dict = {}
        if country in univariate_results:
            for var, lags in univariate_results[country].items():
                relevant_lags = [r['lag'] for r in lags if r['relevant']]
                if relevant_lags:
                    relevant_lags_dict[var] = relevant_lags
        # Always include dlnPD_lag and lnPD_TTC_gap
        macro_vars_mv = []
        for var, lags in relevant_lags_dict.items():
            for lag in lags:
                if lag == 0:
                    macro_vars_mv.append(var)
                elif lag > 0:
                    macro_vars_mv.append(f"{var}_lag{lag}")
                else:
                    macro_vars_mv.append(f"{var}_lead{abs(lag)}")
        macro_vars_mv += ['dlnPD_lag', 'lnPD_TTC_gap']
        results = analyze_country(data, country, macro_vars_mv, output_dir, positive_vars, negative_vars, relevant_lags_dict=relevant_lags_dict)
        if results:
            all_results[country] = results
    # Save detailed results
    with open(output_dir / 'regression_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    # Create summary
    summary_stats = {
        'total_countries': len(all_results),
        'avg_r2': float(np.mean([r['r2'] for r in all_results.values()])) if all_results else 0.0,
        'avg_adj_r2': float(np.mean([r['adj_r2'] for r in all_results.values()])) if all_results else 0.0,
        'variable_frequency': {var: int(sum(1 for r in all_results.values() if var in r['selected_vars'])) 
                             for var in set([v for r in all_results.values() for v in r['selected_vars']])},
        'sign_constraints': {
            'positive_vars': positive_vars,
            'negative_vars': negative_vars
        }
    }
    with open(output_dir / 'regression_summary.txt', 'w') as f:
        f.write("REGRESSION ANALYSIS SUMMARY\n")
        f.write("==========================\n\n")
        f.write(f"Total countries analyzed: {summary_stats['total_countries']}\n")
        f.write(f"Average R-squared: {summary_stats['avg_r2']:.3f}\n")
        f.write(f"Average Adjusted R-squared: {summary_stats['avg_adj_r2']:.3f}\n\n")
        f.write("Variable Selection Frequency:\n")
        f.write("--------------------------\n")
        for var, freq in summary_stats['variable_frequency'].items():
            if summary_stats['total_countries'] > 0:
                f.write(f"{var}: {freq} countries ({freq/summary_stats['total_countries']*100:.1f}%)\n")
            else:
                f.write(f"{var}: {freq} countries\n")
        f.write("\nSign Constraints:\n")
        f.write("----------------\n")
        f.write(f"Positive (expected +): {', '.join(positive_vars)}\n")
        f.write(f"Negative (expected -): {', '.join(negative_vars)}\n")
        f.write("\nDetailed Country Results:\n")
        f.write("----------------------\n")
        for country, results in all_results.items():
            f.write(f"\n{country}:\n")
            f.write(f"  R-squared: {results['r2']:.3f}\n")
            f.write(f"  Adjusted R-squared: {results['adj_r2']:.3f}\n")
            f.write(f"  Selected variables: {', '.join(results['selected_vars'])}\n")
            f.write(f"  Coefficients:\n")
            for var, coef in results['coefficients'].items():
                pval = results['p_values'].get(var, None)
                f.write(f"    {var}: coef={coef:.4f}, p={pval:.4g}\n")
            f.write(f"  Heteroskedasticity test p-value: {results['het_test_p']:.3f}\n")
            f.write(f"  Durbin-Watson statistic: {results['durbin_watson']:.3f}\n")
    print("\nAnalysis complete! Results saved in Output/5.regression_analysis/")

if __name__ == "__main__":
    main() 