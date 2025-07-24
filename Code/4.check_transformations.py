import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from scipy import stats
import sys
import re
import itertools

# Set up logging
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SCENARIOS = ['Baseline', 'S1', 'S3', 'S4']
EXPECTED_TRANS = {
    'GDP': 'log_return',
    'Inflation': 'log_return',  # CPI should use log returns (inflation rate)
    'FX': 'log_return',
    'Government 10Y Bond Rate': 'diff',
    'Unemployment Rate': 'diff',
    'Debt to GDP Ratio': 'diff',
    'Net Exports': 'diff',
    'Equity': 'log_return',
    'Oil Price': 'log_return',
    'Government Consumption': 'diff',
    'Commodity Index': 'log_return',
    'Monetary Policy Rate': 'diff',  # Short-term interest rate
    'Term Spread': 'diff',  # Yield spread (10Y - Monetary Policy Rate)
}

# --- UTILITY FUNCTIONS ---
def remove_seasonality(series, period=4, mask=None):
    """Remove seasonality from a time series using STL decomposition, only using historical data if mask is provided."""
    try:
        if mask is not None:
            # Ensure index is DatetimeIndex and not empty
            if len(series) == 0 or not isinstance(series.index, pd.DatetimeIndex):
                return series, None
            hist_series = series[mask]
        else:
            hist_series = series
        if len(hist_series) == 0 or hist_series.isna().mean() > 0.5:
            return series, None

        # Only handle missing values at the beginning of the series
        first_valid_idx = hist_series.first_valid_index()
        last_hist_idx = hist_series.index[hist_series.index < pd.Timestamp('2025-07-01')][-1] if len(hist_series.index[hist_series.index < pd.Timestamp('2025-07-01')]) > 0 else None
        if first_valid_idx is None or last_hist_idx is None:
            return series, None

        # Get the series from first valid to last historical value
        clean_series = hist_series.loc[first_valid_idx:last_hist_idx]
        clean_series = clean_series.dropna()
        if len(clean_series) < period * 2:
            return series, None

        decomposition = seasonal_decompose(clean_series, period=period, model='additive')
        adjusted = series.copy()
        # Only update the clean segment (first_valid_idx to last_hist_idx)
        seasonal_component = pd.Series(decomposition.seasonal, index=clean_series.index)
        adjusted.loc[clean_series.index] = series.loc[clean_series.index] - seasonal_component
        return adjusted, decomposition
    except Exception as e:
        logger.error(f"Error in seasonal adjustment: {str(e)}")
        return series, None

def apply_ma_detrending_to_variable(data, variable_name, scenarios, ma_window=3):
    """
    Apply moving average detrending to a variable's log returns using previous 3Q MA.
    This follows the pattern: det.ret = ret - lag(SMA(ret, n=3))
    """
    print(f"\nApplying {ma_window}Q moving average detrending to {variable_name} log returns...")
    
    countries_processed = 0
    total_detrended_obs = 0
    
    for scenario in scenarios:
        var_col = f"{variable_name}_{scenario}_trans"
        
        if var_col not in data.columns:
            print(f"Warning: {var_col} not found - skipping scenario {scenario}")
            continue
        
        print(f"Processing {var_col}...")
        
        # Process each country separately
        for country in data['cinc'].unique():
            country_mask = data['cinc'] == country
            country_data = data[country_mask].copy().sort_values('yyyyqq')
            
            if len(country_data) < ma_window + 1:  # Need at least ma_window + 1 observations
                continue
            
            # Get the variable log returns for this country
            var_series = country_data[var_col].copy()
            
            if var_series.isna().all():
                continue
            
            # Calculate 3Q moving average (using previous 3 quarters)
            # This creates a lagged moving average: MA[t] = mean(ret[t-2], ret[t-1], ret[t])
            ma_series = var_series.rolling(window=ma_window, min_periods=ma_window).mean()
            
            # Lag the MA by 1 period: lag(SMA(ret, n=3))
            # This means we use MA[t-1] to detrend ret[t]
            lagged_ma = ma_series.shift(1)
            
            # Calculate detrended returns: det.ret = ret - lag(SMA(ret, n=3))
            detrended_series = var_series - lagged_ma
            
            # Count valid detrended observations
            valid_detrended = detrended_series.notna().sum()
            if valid_detrended > 0:
                total_detrended_obs += valid_detrended
                
                # Update the data with detrended values
                data.loc[country_mask, var_col] = detrended_series.values
        
        countries_processed = data['cinc'].nunique()
    
    print(f"MA detrending applied to {variable_name}:")
    print(f"  Countries processed: {countries_processed}")
    print(f"  Total detrended observations: {total_detrended_obs}")
    print(f"  MA window: {ma_window} quarters (lagged)")
    print(f"  Method: det.ret = ret - lag(SMA(ret, n={ma_window}))")

def check_stationarity(series, alpha=0.05):
    """Check stationarity using ADF test."""
    try:
        clean_series = series.dropna()
        if len(clean_series) < 20:
            return {'is_stationary': False, 'p_value': np.nan, 'error': 'Insufficient data'}
        result = adfuller(clean_series, regression='ct')
        return {
            'is_stationary': result[1] < alpha,
            'p_value': result[1],
            'error': None
        }
    except Exception as e:
        return {'is_stationary': False, 'p_value': np.nan, 'error': str(e)}

def safe_log_return(series, epsilon=1e-6):
    """Scale series by its mean before log return to treat extremely small values. This does not affect log return results, but improves numerical stability."""
    if len(series.dropna()) == 0:
        return series
    mean_val = np.nanmean(series)
    if mean_val == 0 or np.isnan(mean_val):
        # Fallback: just use epsilon
        scaled = series + epsilon
    else:
        scaled = series / mean_val
    # Add epsilon to avoid log(0)
    return np.log1p(scaled + epsilon)

def decide_transformation(var, baseline_col, data):
    """Decide transformation for a macro variable using only Baseline and historical data."""
    print(f"Deciding transformation for {var}")
    mask_hist = data['yyyyqq'] < pd.Timestamp('2025-07-01')
    stationarity_results = []
    seasonality_results = []
    countries = data['cinc'].unique()
    
    # First check if any values are negative
    has_negative = False
    for country in countries:
        country_data = data[(data['cinc'] == country) & mask_hist].copy()
        if len(country_data) == 0:
            continue
        if (country_data[baseline_col] <= 0).any():
            has_negative = True
            break
    
    # If any values are negative, only consider 'diff' transformation
    if has_negative:
        logger.info(f"{var} has negative values - using 'diff' transformation only")
        transformations_to_try = ['original', 'diff']
    else:
        transformations_to_try = ['original', 'diff', 'log_return']
    
    for country in countries:
        country_data = data[(data['cinc'] == country) & mask_hist].copy()
        if len(country_data) == 0:
            continue
        country_data = country_data.set_index('yyyyqq')
        series = country_data[baseline_col]
        if len(series) == 0:
            continue
        adjusted_series, decomp = remove_seasonality(series, mask=(series.index < pd.Timestamp('2025-07-01')) if isinstance(series.index, pd.DatetimeIndex) else None)
        if decomp is not None:
            try:
                plt.figure(figsize=(12, 10))
                decomp.plot()
                plt.title(f'Seasonal Decomposition - {var} - {country}')
                plt.tight_layout()
                plt.savefig(f'Output/4.transformation/seasonal_decomp_{country}_{var}.png')
                plt.close('all')
                trend_std = np.std(decomp.trend)
                if trend_std and not np.isnan(trend_std):
                    seasonality_strength = np.std(decomp.seasonal) / trend_std
                else:
                    seasonality_strength = np.nan
                seasonality_results.append(seasonality_strength)
            except Exception as e:
                logger.error(f"Error plotting seasonal decomposition for {var} - {country}: {str(e)}")
                plt.close('all')
        
        for trans_name in transformations_to_try:
            if trans_name == 'original':
                trans_series = adjusted_series
            elif trans_name == 'diff':
                trans_series = adjusted_series.diff()
            elif trans_name == 'log_return':
                # Scale by mean to treat extremely small values (does not affect log return result)
                trans_series = safe_log_return(adjusted_series).diff()
            
            if trans_series is not None:
                stat_test = check_stationarity(trans_series)
                if stat_test['error'] is None:
                    stationarity_results.append({
                        'country': country,
                        'transformation': trans_name,
                        'is_stationary': stat_test['is_stationary'],
                        'p_value': stat_test['p_value']
                    })
    
    var_results = {'stationarity_tests': {}, 'seasonality_tests': {}, 'distribution_tests': {}}
    if stationarity_results:
        stationarity_df = pd.DataFrame(stationarity_results)
        var_results['stationarity_tests'] = {
            'stationary_ratio': stationarity_df.groupby('transformation')['is_stationary'].mean().to_dict(),
            'mean_p_value': stationarity_df.groupby('transformation')['p_value'].mean().to_dict()
        }
    if seasonality_results:
        var_results['seasonality_tests'] = {
            'mean_seasonality_strength': np.nanmean(seasonality_results),
            'std_seasonality_strength': np.nanstd(seasonality_results)
        }
    
    # --- Concise transformation decision logic ---
    if var_results['stationarity_tests']:
        stationary_ratios = var_results['stationarity_tests']['stationary_ratio']
        mean_p_values = var_results['stationarity_tests']['mean_p_value']
        
        # If any values are negative, only diff is allowed
        if has_negative:
            best_trans = 'diff'
            reason = "Variable has negative values - only diff transformation possible"
        else:
            diff_ratio = stationary_ratios.get('diff', 0)
            logret_ratio = stationary_ratios.get('log_return', 0)
            # If both are equally good (within 1%), use expected transformation
            if abs(diff_ratio - logret_ratio) < 0.01:
                best_trans = EXPECTED_TRANS.get(var, 'diff')
                reason = f"Similar performance (diff: {diff_ratio:.1%}, log_return: {logret_ratio:.1%}) - using expected transformation"
            else:
                best_trans = 'diff' if diff_ratio > logret_ratio else 'log_return'
                reason = f"Better stationarity: {best_trans} ({(diff_ratio if best_trans == 'diff' else logret_ratio):.1%} vs {(logret_ratio if best_trans == 'diff' else diff_ratio):.1%})"
        
        var_results['recommended_transformation'] = best_trans
        var_results['reason'] = reason
        var_results['stationarity_pvalue'] = mean_p_values.get(best_trans, np.nan)
        var_results['has_negative_values'] = has_negative
    else:
        # Fallback: use expected transformation or diff
        best_trans = 'diff' if has_negative else EXPECTED_TRANS.get(var, 'diff')
        var_results['recommended_transformation'] = best_trans
        var_results['reason'] = "No valid stationarity tests - using fallback transformation"
        var_results['stationarity_pvalue'] = np.nan
        var_results['has_negative_values'] = has_negative
    
    # Add seasonality information
    if var_results['seasonality_tests']:
        seasonality_strength = var_results['seasonality_tests'].get('mean_seasonality_strength', np.nan)
        var_results['seasonality_detected'] = not pd.isna(seasonality_strength) and seasonality_strength > 0.1
    else:
        var_results['seasonality_detected'] = False
    
    return var_results

def apply_seasonal_adjustment(data, var, scenarios, mask_hist):
    """Apply seasonal adjustment to a variable across all scenarios"""
    baseline_col = f"{var}_Baseline"
    
    for country in data['cinc'].unique():
        mask_country_hist = (data['cinc'] == country) & mask_hist
        hist_series = data.loc[mask_country_hist, baseline_col]
        
        # Only apply seasonality if index is a non-empty DatetimeIndex
        if len(hist_series) > 0 and isinstance(hist_series.index, pd.DatetimeIndex):
            hist_mask = hist_series.index < pd.Timestamp('2025-07-01')
            adjusted_hist, decomp = remove_seasonality(hist_series, mask=hist_mask)
        else:
            adjusted_hist, decomp = hist_series, None
            
        mask_country = (data['cinc'] == country)
        mask_country_hist = mask_country & mask_hist
        
        if decomp is not None:
            # Update baseline historical with adjusted data
            data.loc[mask_country_hist, baseline_col] = adjusted_hist
            
            # Copy adjusted historical data to other scenarios
            for scen in scenarios[1:]:  # Skip baseline
                scen_col = f"{var}_{scen}"
                if scen_col in data.columns:
                    data.loc[mask_country_hist, scen_col] = adjusted_hist[mask_country_hist]

def apply_transformation_to_variable(data, var, scenarios, trans_type):
    """Apply transformation to a variable across all scenarios"""
    for scen in scenarios:
        col = f"{var}_{scen}"
        if col in data.columns:
            series = data[col]
            # Create transformed column with NaN values
            data[f'{col}_trans'] = pd.Series(index=series.index, dtype=float)
            
            # Only transform where we have valid values
            mask_valid = ~series.isna()
            if trans_type == 'diff':
                data.loc[mask_valid, f'{col}_trans'] = series[mask_valid].diff()
            elif trans_type == 'log_return':
                # Scale by mean to treat extremely small values (does not affect log return result)
                data.loc[mask_valid, f'{col}_trans'] = safe_log_return(series[mask_valid]).diff()
            else:
                data.loc[mask_valid, f'{col}_trans'] = series[mask_valid]

def create_term_spread_variable(data, scenarios):
    """Create Term Spread variable (10Y Bond Rate - Monetary Policy Rate)"""
    term_spread_var = "Term Spread"
    
    print(f"Creating {term_spread_var} variable...")
    
    # Ensure yyyyqq is datetime
    if data['yyyyqq'].dtype == 'object':
        data['yyyyqq'] = pd.to_datetime(data['yyyyqq'], errors='coerce')
    
    for scenario in scenarios:
        bond_col = f"Government 10Y Bond Rate_{scenario}"
        mmr_col = f"Monetary Policy Rate_{scenario}"  # Fixed column name
        spread_col = f"{term_spread_var}_{scenario}"
        
        if bond_col in data.columns and mmr_col in data.columns:
            # Initialize the spread column
            data[spread_col] = np.nan
            
            # Calculate spread for all rows where both rates are available
            # Use vectorized operation instead of country-by-country loop
            valid_mask = (
                data[bond_col].notna() & 
                data[mmr_col].notna() & 
                data['yyyyqq'].notna()  # Ensure valid dates
            )
            
            if valid_mask.any():
                data.loc[valid_mask, spread_col] = (
                    data.loc[valid_mask, bond_col] - data.loc[valid_mask, mmr_col]
                )
                
                valid_count = valid_mask.sum()
                countries_with_spread = data.loc[valid_mask, 'cinc'].nunique()
                print(f"Created {spread_col}: {valid_count} valid observations across {countries_with_spread} countries")
            else:
                print(f"Warning: No valid observations for {spread_col} - no overlapping data")
        else:
            print(f"Warning: Cannot create {spread_col} - missing {bond_col} or {mmr_col}")

def apply_transformations(data, macro_vars, scenarios, transformation_decisions):
    """Apply transformation and seasonality to all scenario columns for each variable."""
    transformed_data = data.copy()
    mask_hist = transformed_data['yyyyqq'] < pd.Timestamp('2025-07-01')
    
    # Create Term Spread variable (10Y Bond Rate - Monetary Policy Rate)
    print("Creating Term Spread variable (10Y Bond Rate - Monetary Policy Rate)...")
    create_term_spread_variable(transformed_data, scenarios)
    
    # Add Term Spread to macro_vars for transformation
    enhanced_macro_vars = macro_vars.copy()
    term_spread_cols = [f"Term Spread_{scen}" for scen in scenarios 
                       if f"Term Spread_{scen}" in transformed_data.columns]
    if term_spread_cols:
        enhanced_macro_vars["Term Spread"] = term_spread_cols
    
    # Process all variables (original + Term Spread)
    print("Applying seasonal adjustments and transformations...")
    for var in enhanced_macro_vars:
        # Determine transformation type
        if var == 'Term Spread':
            trans_type = EXPECTED_TRANS.get(var, 'diff')
        else:
            trans_type = transformation_decisions[var]['recommended_transformation']
        
        # Apply seasonal adjustment
        apply_seasonal_adjustment(transformed_data, var, scenarios, mask_hist)
        
        # Apply transformation
        apply_transformation_to_variable(transformed_data, var, scenarios, trans_type)
                            
    return transformed_data

def plot_single_variable_transformation(data, transformed_data, var, output_dir):
    """Plot transformation results for a single variable"""
    baseline_col = f'{var}_Baseline'
    trans_var = f'{baseline_col}_trans'
    
    # Skip if columns don't exist
    if baseline_col not in data.columns:
        return
        
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original Data (Baseline)
    plt.subplot(2, 2, 1)
    for country in data['cinc'].unique()[:5]:
        country_data = data[data['cinc'] == country]
        plt.plot(country_data['yyyyqq'], country_data[baseline_col], label=country, alpha=0.5)
    
    plt.title(f'{var} (Baseline) - Original Data (Sample Countries)')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 2: Transformed Data (Baseline)
    plt.subplot(2, 2, 2)
    if trans_var in transformed_data.columns:
        for country in transformed_data['cinc'].unique()[:5]:
            country_data = transformed_data[transformed_data['cinc'] == country]
            plt.plot(country_data['yyyyqq'], country_data[trans_var], label=country, alpha=0.5)
        plt.title(f'{var} (Baseline) - Transformed Data (Sample Countries)')
        plt.xticks(rotation=45)
        plt.legend()
    
    # Plot 3: Distribution of Transformed Data (Baseline)
    plt.subplot(2, 2, 3)
    if trans_var in transformed_data.columns:
        sns.histplot(data=transformed_data, x=trans_var, bins=50)
        plt.title(f'{var} (Baseline) - Distribution of Transformed Data')
    
    # Plot 4: QQ Plot (Baseline)
    plt.subplot(2, 2, 4)
    if trans_var in transformed_data.columns:
        stats.probplot(transformed_data[trans_var].dropna(), dist="norm", plot=plt)
        plt.title(f'{var} (Baseline) - QQ Plot')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'transformation_{var}.png')
    plt.close()

def plot_transformation_results(data, transformed_data, macro_vars, scenarios, output_dir):
    """Plot transformation results for each variable (Baseline only)."""
    # Plot original macro variables
    for var in macro_vars:
        plot_single_variable_transformation(data, transformed_data, var, output_dir)
    
    # Plot Term Spread variable
    plot_single_variable_transformation(data, transformed_data, 'Term Spread', output_dir)

def test_final_stationarity(transformed_data, macro_vars, transformation_decisions, output_dir):
    """Test stationarity of final transformed macro variables (Baseline + historical data only)"""
    print("\n" + "="*60)
    print("TESTING FINAL TRANSFORMED VARIABLES STATIONARITY")
    print("="*60)
    print("Testing Baseline scenario with historical data only (same as transformation decision process)")
    
    # Variables that underwent MA detrending
    ma_detrended_vars = ['Inflation', 'Equity', 'Government Consumption']
    
    # Get historical data mask
    mask_hist = transformed_data['yyyyqq'] < pd.Timestamp('2025-07-01')
    
    final_stationarity_results = []
    summary_results = {}
    
    # Test all macro variables plus Term Spread
    all_vars = list(macro_vars.keys()) + ['Term Spread']
    
    for var in all_vars:
        print(f"\n--- Testing {var} (Baseline + Historical) ---")
        
        # Get transformation info
        trans_info = transformation_decisions.get(var, {})
        recommended_trans = trans_info.get('recommended_transformation', 'unknown')
        has_ma_detrending = var in ma_detrended_vars
        
        # Only test Baseline scenario
        trans_col = f"{var}_Baseline_trans"
        
        if trans_col not in transformed_data.columns:
            print(f"  Skipping: column {trans_col} not found")
            continue
        
        scenario_results = []
        countries_tested = 0
        countries_passed = 0
        
        # Test each country (historical data only)
        for country in transformed_data['cinc'].unique():
            country_mask = (transformed_data['cinc'] == country) & mask_hist
            country_data = transformed_data[country_mask]
            
            if len(country_data) == 0:
                continue
            
            series = country_data[trans_col].dropna()
            
            if len(series) < 20:  # Need sufficient data for ADF test
                continue
            
            countries_tested += 1
            
            # Perform stationarity test
            stat_result = check_stationarity(series)
            
            if stat_result['error'] is None:
                is_stationary = stat_result['is_stationary']
                p_value = stat_result['p_value']
                
                if is_stationary:
                    countries_passed += 1
                
                scenario_results.append({
                    'variable': var,
                    'country': country,
                    'transformation': recommended_trans,
                    'ma_detrended': has_ma_detrending,
                    'is_stationary': is_stationary,
                    'p_value': p_value,
                    'observations': len(series)
                })
                
                final_stationarity_results.append({
                    'variable': var,
                    'country': country,
                    'transformation': recommended_trans,
                    'ma_detrended': has_ma_detrending,
                    'is_stationary': is_stationary,
                    'p_value': p_value,
                    'observations': len(series)
                })
        
        # Calculate summary statistics
        if countries_tested > 0:
            pass_rate = countries_passed / countries_tested
            avg_p_value = np.mean([r['p_value'] for r in scenario_results])
            
            print(f"  {countries_passed}/{countries_tested} countries passed ({pass_rate:.1%}) | Avg p-value: {avg_p_value:.4f}")
            
            if pass_rate >= 0.80:
                status = "✅ Excellent"
            elif pass_rate >= 0.70:
                status = "✅ Good"
            elif pass_rate >= 0.60:
                status = "⚠️  Acceptable"
            else:
                status = "❌ Poor"
            
            print(f"  Status: {status}")
            
            var_summary = {
                'variable': var,
                'transformation': recommended_trans,
                'ma_detrended': has_ma_detrending,
                'countries_tested': countries_tested,
                'countries_passed': countries_passed,
                'pass_rate': pass_rate,
                'avg_p_value': avg_p_value,
                'status': status
            }
        else:
            print(f"  No countries with sufficient data")
            var_summary = {
                'variable': var,
                'transformation': recommended_trans,
                'ma_detrended': has_ma_detrending,
                'countries_tested': 0,
                'countries_passed': 0,
                'pass_rate': 0.0,
                'avg_p_value': np.nan,
                'status': "❌ No Data"
            }
        
        summary_results[var] = var_summary
    
    # Save detailed results to CSV
    if final_stationarity_results:
        results_df = pd.DataFrame(final_stationarity_results)
        results_file = f'{output_dir}/final_stationarity_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed stationarity results saved to: {results_file}")
        
        # Save summary results to JSON
        summary_file = f'{output_dir}/final_stationarity_summary.json'
        with open(summary_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return obj
            
            json_summary = {}
            for var, data in summary_results.items():
                json_summary[var] = {}
                for key, value in data.items():
                    json_summary[var][key] = convert_numpy(value)
            
            json.dump(json_summary, f, indent=2)
        
        print(f"Summary results saved to: {summary_file}")
        
        # Print overall summary
        print("\n" + "="*60)
        print("FINAL STATIONARITY SUMMARY (Baseline + Historical)")
        print("="*60)
        
        baseline_summary = []
        for var, result in summary_results.items():
            baseline_summary.append({
                'variable': var,
                'transformation': result['transformation'],
                'ma_detrended': result['ma_detrended'],
                'pass_rate': result['pass_rate'],
                'status': result['status']
            })
        
        if baseline_summary:
            print("\nResults:")
            for result in sorted(baseline_summary, key=lambda x: x['pass_rate'], reverse=True):
                ma_indicator = " (MA detrended)" if result['ma_detrended'] else ""
                print(f"  {result['variable']:20} | {result['transformation']:10} | {result['pass_rate']:5.1%} | {result['status']}{ma_indicator}")
        
        # Count problematic variables
        poor_vars = [r for r in baseline_summary if r['pass_rate'] < 0.70]
        if poor_vars:
            print(f"\n⚠️  Variables with concerning pass rates (<70%): {len(poor_vars)}")
            for var in poor_vars:
                print(f"    - {var['variable']} ({var['pass_rate']:.1%})")
        else:
            print(f"\n✅ All variables achieved good stationarity pass rates (≥70%)")
            
        print(f"\nTotal variables tested: {len(summary_results)}")
        print(f"Total country-variable combinations: {len(final_stationarity_results)}")
    
    else:
        print("No stationarity results to save!")

def create_transformation_summary(transformation_decisions, output_dir):
    """Save transformation decisions to a summary file"""
    summary_file = f'{output_dir}/transformation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("TRANSFORMATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        for var, decision in transformation_decisions.items():
            if isinstance(decision, dict) and 'recommended_transformation' in decision:
                f.write(f"Variable: {var}\n")
                f.write(f"  Transformation: {decision['recommended_transformation']}\n")
                
                # Format reason
                reason = decision.get('reason', 'N/A')
                f.write(f"  Reason: {reason}\n")
                
                # Format stationarity p-value
                p_value = decision.get('stationarity_pvalue', np.nan)
                if pd.isna(p_value):
                    p_value_str = "N/A"
                else:
                    p_value_str = f"{p_value:.4f}"
                f.write(f"  Stationarity p-value: {p_value_str}\n")
                
                # Format seasonality detection
                seasonality = decision.get('seasonality_detected', False)
                seasonality_str = "Yes" if seasonality else "No"
                f.write(f"  Seasonality detected: {seasonality_str}\n")
                
                # Add negative values indicator
                has_negative = decision.get('has_negative_values', False)
                f.write(f"  Has negative values: {'Yes' if has_negative else 'No'}\n")
                
                # Add stationarity test results if available
                if 'stationarity_tests' in decision and decision['stationarity_tests']:
                    stat_results = decision['stationarity_tests']
                    if 'stationary_ratio' in stat_results:
                        f.write(f"  Stationarity pass rates:\n")
                        for trans_type, ratio in stat_results['stationary_ratio'].items():
                            f.write(f"    {trans_type}: {ratio:.1%}\n")
                
                f.write("\n")
    print(f"Transformation summary saved to: {summary_file}")

def main():
    try:
        # --- Load regression data (contains both PD and macro variables) ---
        print("Loading regression data with seasonally adjusted PD...")
        data = pd.read_csv('Output/3.regression_data.csv')
        data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
        
        output_dir = 'Output/4.transformation'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # --- Check PD seasonal adjustment status ---
        if 'cdsiedf5_sa_method' in data.columns:
            sa_summary = data.groupby('cinc')['cdsiedf5_sa_method'].first().value_counts()
            print(f"\nPD Seasonal Adjustment Status (from Step 3):")
            for method, count in sa_summary.items():
                print(f"  {method}: {count} countries")
            
            total_sa_obs = (data['cdsiedf5_sa_method'] != 'none').sum()
            print(f"Total observations with seasonal adjustment: {total_sa_obs}")
        else:
            print("\nNo seasonal adjustment information found - using original PD data")
        
        # --- Apply transformations to macro variables ---
        print("\n" + "="*60)
        print("STEP 2: MACRO VARIABLE TRANSFORMATIONS")
        print("="*60)
        
        macro_pattern = re.compile(r'^(.*)_(Baseline|S1|S3|S4)$')
        macro_vars = {}
        for col in data.columns:
            m = macro_pattern.match(col)
            if m:
                var, scen = m.groups()
                macro_vars.setdefault(var, []).append(col)
        
        print(f"Detected {len(macro_vars)} macro variables across {len(SCENARIOS)} scenarios")
        
        # --- Decide transformations using Baseline/historical data ---
        transformation_decisions = {}
        for var in macro_vars:
            baseline_col = f"{var}_Baseline"
            var_results = decide_transformation(var, baseline_col, data)
            transformation_decisions[var] = var_results
        
        # Add Term Spread to transformation decisions (will be created during transformation)
        transformation_decisions['Term Spread'] = {
            'recommended_transformation': EXPECTED_TRANS.get('Term Spread', 'diff'),
            'stationarity_tests': {},
            'seasonality_tests': {},
            'distribution_tests': {}
        }
        
        # --- Apply transformations to all scenarios ---
        transformed_data = apply_transformations(data, macro_vars, SCENARIOS, transformation_decisions)
        
        # --- Apply 3Q MA detrending to specific variables ---
        print("\n" + "="*60)
        print("STEP 3: MA DETRENDING FOR SELECTED VARIABLES")
        print("="*60)
        
        # Apply 3Q moving average detrending to Inflation, Equity, and Government Consumption log returns
        # This follows the R code pattern: det.ret = ret - lag(SMA(ret, n=3))
        variables_to_detrend = ['Inflation', 'Equity', 'Government Consumption']
        
        for var in variables_to_detrend:
            apply_ma_detrending_to_variable(transformed_data, var, SCENARIOS, ma_window=3)
        
        # --- Test final stationarity ---
        print("\n" + "="*60)
        print("STEP 4: TESTING FINAL STATIONARITY")
        print("="*60)
        
        # Test stationarity of final transformed variables (including MA detrended ones)
        test_final_stationarity(transformed_data, macro_vars, transformation_decisions, output_dir)
        
        # --- Save final transformed data ---
        print("\n" + "="*60)
        print("STEP 5: SAVING TRANSFORMED DATA")
        print("="*60)
        
        # Save the transformed data
        transformed_data.to_csv(f'{output_dir}/transformed_data.csv', index=False)
        print(f"Transformed data saved to: {output_dir}/transformed_data.csv")
        
        # Create summary report
        create_transformation_summary(transformation_decisions, output_dir)
        
        print(f"\nTransformation complete! Results saved in: {output_dir}")
        print(f"Total countries: {transformed_data['cinc'].nunique()}")
        print(f"Total observations: {len(transformed_data)}")
        
        # Print final column summary
        print(f"\nFinal dataset columns ({len(transformed_data.columns)}):")
        baseline_cols = [col for col in transformed_data.columns if col.endswith('_Baseline_trans')]
        scenario_cols = [col for col in transformed_data.columns if any(col.endswith(f'_{s}_trans') for s in ['S1', 'S3', 'S4'])]
        other_cols = [col for col in transformed_data.columns if col not in baseline_cols + scenario_cols]
        
        print(f"  Baseline transformed variables: {len(baseline_cols)}")
        print(f"  Scenario transformed variables: {len(scenario_cols)}")
        print(f"  Other columns: {len(other_cols)}")

    except Exception as e:
        print(f"Error in transformation process: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        plt.close('all')

if __name__ == "__main__":
    main() 