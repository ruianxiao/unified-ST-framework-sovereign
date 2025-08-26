"""
OVS (Optimal Variable Selection) for Sovereign Credit Risk Models

This script performs optimal variable selection for sovereign credit risk models with the following features:

1. Historical Data Processing: Processes historical sovereign credit data with winsorization and outlier handling
   - TEMPORARY FILTER: Excludes Canada PD data before 2009 Q1 due to data quality issues
2. GCorr Forecast Integration: Optionally includes GCorr forecast scenarios (S1, S3, S4, baseline) for additional training data
   - GCorr quarterly PD forecasts are converted to annual PD using survival probability formula: PD_annual = 1 - (1 - PD_quarterly)^4
3. Variable Selection: Systematically tests combinations of macro variables with different lag structures
4. Model Constraints: Applies sign constraints and statistical criteria (p-values, correlation limits)
5. Parallel Processing: Uses multiprocessing for efficient country-level model selection

Key Configuration Options:
- INCLUDE_GCORR_FORECAST: Enable/disable inclusion of GCorr forecast data (default: False)
- GCORR_FORECAST_QUARTERS: Number of forecast quarters to include (default: 40, max: 122)
- GCORR_USE_SMOOTHED_PD: Use smoothed PD values from GCorr (default: True)
- DATA_PREPARATION_ONLY: Exit after data preparation without running variable selection (default: False)
- SAVE_PREPARED_DATA: Save prepared dataset to CSV file (default: False)
- TEST_MODE: Enable for testing on single country
- APPLY_WINSORIZATION: Enable/disable outlier winsorization

Output Files:
- ovs_results.csv: Consolidated results across all countries
- ovs_summary.json: Summary statistics and configuration

When GCorr forecast is enabled, file names include "_gcorr{N}q" suffix to distinguish from historical-only models.
When smoothed PD is used, file names include "_smoothed" suffix to distinguish from unsmoothed results.

Data Consistency Notes:
- Historical data: Annual PD values from CDS spreads (cdsiedf5)
- GCorr forecast data: Quarterly PD converted to annual using survival probability formula
- This ensures seamless integration between historical and forecast data periods

GCorr Forecast Integration Details:
- When INCLUDE_GCORR_FORECAST=True, historical data is duplicated for each scenario (S1, S3, S4, baseline)
- Each scenario gets its own complete trajectory: historical data + scenario-specific forecasts
- dlnPD calculations are performed within each country-scenario trajectory to ensure continuity
- The first forecast dlnPD = ln(first_forecast_PD) - ln(last_historical_PD) for each scenario
- This allows proper transition from historical to forecast periods without discontinuities
- All scenarios share the same historical data but diverge in the forecast period
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations, product
from pathlib import Path
# Plotting imports removed - not used in this script
import json
import concurrent.futures
import sys

# --- CONFIGURATION ---
DATA_PATH = 'Output/4.transformation/transformed_data.csv'
OUTPUT_DIR = Path('Output/6.ovs_variable_selection')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- VARIABLE DEFINITIONS ---
country_col = 'cinc'
depvar = 'dlnPD'  # log change in PIT PD
ttc_col = 'TTC_PD'  # to be computed as historical average PD per country
lag_col = 'dlnPD_lag1'  # to be created AFTER winsorization
meanrev_col = 'mean_reverting'  # to be created

# --- OVS PARAMETERS ---
max_k = 4  # max number of macro variables (excluding lag and mean-reverting)
max_lag = 0  # allow 0-4 lags per macro variable
max_corr = 0.8
max_pval = 0.3

# --- SIGN CONSTRAINTS ---
# Map macro variable base names to sign constraints based on scenario ordering analysis
macro_var_sign_map = {
    # Positive coefficients (stress worsens these variables)
    'Debt to GDP Ratio': 'positive',
    'Unemployment Rate': 'positive',
    
    # Negative coefficients (stress improves/lowers these variables)
    'Commodity Index': 'negative',
    'Equity': 'negative', 
    'GDP': 'negative',
    'Oil Price': 'negative',
    'Government Consumption': 'negative',
    'Inflation': 'negative',
    
    # Flexible (country-specific direction)
    'FX': 'flexible',
    'Net Exports': 'flexible', 
    'Government 10Y Bond Rate': 'flexible',
    'Monetary Policy Rate': 'flexible',
    'Term Spread': 'flexible'
}

# --- TEST CONFIGURATION ---
TEST_MODE = False  # Set to True to run on just one country for testing
TEST_COUNTRY = 'USA'  # Country to test with

# --- OUTLIER HANDLING CONFIGURATION ---
STD_MULTIPLIER = 2.5  # Standard deviations for winsorization bounds
APPLY_WINSORIZATION = False  # Enable/disable winsorization

# --- GCORR FORECAST CONFIGURATION ---
INCLUDE_GCORR_FORECAST = True  # Set to True to include GCorr forecast data in regression
GCORR_FORECAST_QUARTERS = 40  # Number of forecast quarters to include (max 122)
GCORR_USE_SMOOTHED_PD = True  # Set to True to use smoothed PD, False for unsmoothed PD
GCORR_FORECAST_DIR = Path('gcorr-research-delivery-validation/Output/gcorr_scenario_plots/annualized_data')

# --- DATA PREPARATION MODE ---
DATA_PREPARATION_ONLY = False  # Set to True to run only data preparation and save results
SAVE_PREPARED_DATA = True  # Set to True to save prepared data to CSV

def winsorize_dlnPD(data, std_multiplier=2.5):
    """
    Simple winsorization function that only applies to dlnPD.
    Returns the data with winsorized dlnPD and summary statistics.
    """
    if not APPLY_WINSORIZATION:
        return data.copy(), {}
    
    data_clean = data.copy()
    winsorization_summary = []
    
    for country in data_clean[country_col].unique():
        country_mask = data_clean[country_col] == country
        country_data = data_clean.loc[country_mask, depvar]
        
        if len(country_data.dropna()) < 10:
            continue
            
        mean_val = country_data.mean()
        std_val = country_data.std()
    
        if std_val == 0 or pd.isna(std_val):
            continue
        
        lower_bound = mean_val - (std_multiplier * std_val)
        upper_bound = mean_val + (std_multiplier * std_val)
        
        # Count outliers before winsorization
        outliers_lower = (country_data < lower_bound).sum()
        outliers_upper = (country_data > upper_bound).sum()
        total_outliers = outliers_lower + outliers_upper
        
        if total_outliers > 0:
            # Apply winsorization (truncation)
            data_clean.loc[country_mask, depvar] = country_data.clip(lower=lower_bound, upper=upper_bound)
            
            # Record winsorization details
            winsorization_summary.append({
                'country': country,
                'variable': depvar,
                'outliers_winsorized': total_outliers,
                'outliers_lower': outliers_lower,
                'outliers_upper': outliers_upper,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std_multiplier': std_multiplier
            })
    
    return data_clean, winsorization_summary

def load_gcorr_forecast_data():
    """Load GCorr forecast data from annualized data directory"""
    if not INCLUDE_GCORR_FORECAST:
        return None
    
    try:
        # Load all country files from annualized data directory
        if not GCORR_FORECAST_DIR.exists():
            print(f"Warning: GCorr forecast directory not found: {GCORR_FORECAST_DIR}")
            return None
        
        gcorr_data = []
        file_type = 'smoothed' if GCORR_USE_SMOOTHED_PD else 'unsmoothed'
        
        for file_path in GCORR_FORECAST_DIR.glob(f'*_annualized_{file_type}_pd.csv'):
            try:
                country_data = pd.read_csv(file_path)
                country_data['date'] = pd.to_datetime(country_data['date'])
                
                # Extract country from filename
                country = file_path.stem.replace(f'_annualized_{file_type}_pd', '')
                country_data['country_iso'] = country
                
                gcorr_data.append(country_data)
                
            except Exception as e:
                print(f"Warning: Could not load GCorr file {file_path.name}: {e}")
                continue
        
        if gcorr_data:
            combined_gcorr = pd.concat(gcorr_data, ignore_index=True)
            
            # Filter to only include the specified number of forecast quarters
            forecast_start = pd.to_datetime('2025-07-01')
            forecast_end = forecast_start + pd.DateOffset(months=3*GCORR_FORECAST_QUARTERS)
            
            # Apply quarter filtering
            original_count = len(combined_gcorr)
            combined_gcorr = combined_gcorr[
                (combined_gcorr['date'] >= forecast_start) & 
                (combined_gcorr['date'] <= forecast_end)
            ].copy()
            
            filtered_count = len(combined_gcorr)
            print(f"GCorr annualized data loaded: {original_count} total records from {len(gcorr_data)} countries")
            print(f"Filtered to {GCORR_FORECAST_QUARTERS} quarters ({forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')}): {filtered_count} records")
            
            return combined_gcorr if not combined_gcorr.empty else None
        else:
            print("Warning: No GCorr forecast data files found")
            return None
            
    except Exception as e:
        print(f"Warning: Could not load GCorr forecast data: {e}")
        return None

def create_gcorr_forecast_sample(gcorr_data, historical_data):
    """Create forecast sample using GCorr annualized PD forecasts + transformed data scenario-specific macro variables
    
    This function:
    1. Uses existing transformed data which already contains scenario-specific macro variables
    2. Replaces forecast period PD values with GCorr annualized forecasts
    3. Ensures proper continuity between historical and forecast periods
    """
    if gcorr_data is None:
        return historical_data
    
    # Load the full transformed data which should contain scenario-specific forecasts
    try:
        full_transformed_data = pd.read_csv(DATA_PATH)
        full_transformed_data['yyyyqq'] = pd.to_datetime(full_transformed_data['yyyyqq'])
        
        # Filter for forecast period (>= 2025-07-01)
        forecast_start = pd.to_datetime('2025-07-01')
        forecast_data = full_transformed_data[full_transformed_data['yyyyqq'] >= forecast_start].copy()
        
        if forecast_data.empty:
            return create_gcorr_forecast_sample_pd_only(gcorr_data, historical_data)
        
        # Create historical data expanded for all scenarios
        historical_expanded = []
        scenarios = ['S1', 'S3', 'S4', 'baseline']
        historical_countries = set(historical_data['cinc'].unique())
        
        for country in historical_countries:
            country_hist = historical_data[historical_data['cinc'] == country].copy()
            if country_hist.empty:
                continue
                
            # For each scenario, create a copy of historical data with baseline macro variables
            for scenario in scenarios:
                country_hist_scenario = country_hist.copy()
                country_hist_scenario['scenario'] = scenario
                country_hist_scenario['is_forecast'] = False
                historical_expanded.append(country_hist_scenario)
        
        expanded_historical_df = pd.concat(historical_expanded, ignore_index=True) if historical_expanded else pd.DataFrame()
        
        # Now integrate GCorr PD forecasts with transformed data scenario forecasts
        enhanced_forecast_records = []
        
        # Create mapping from GCorr data for PD updates
        gcorr_updates = {}  # Dictionary to store PD updates by country-scenario-date
        
        # Process each GCorr forecast record
        for _, gcorr_row in gcorr_data.iterrows():
            country_iso = gcorr_row['country_iso']
            scenario = gcorr_row['scenario']
            date = gcorr_row['date']
            annual_pd = gcorr_row['annual_pd']
            
            if country_iso not in historical_countries:
                continue
            
            if pd.isna(annual_pd) or annual_pd <= 0:
                continue
            
            # Store the update
            key = (country_iso, scenario, date)
            gcorr_updates[key] = annual_pd
        
        # Now update forecast data with GCorr PD values while keeping scenario-specific macro variables
        forecast_updated = []
        
        for _, forecast_row in forecast_data.iterrows():
            country = forecast_row['cinc']
            date = forecast_row['yyyyqq']
            
            # Create records for each scenario
            for scenario in scenarios:
                updated_row = forecast_row.copy()
                updated_row['scenario'] = scenario
                updated_row['is_forecast'] = True
                
                # Update baseline macro variables to scenario-specific ones
                for col_name in updated_row.index:
                    if isinstance(col_name, str):  # Fix linter error: ensure col is string
                        if '_Baseline' in col_name and not col_name.endswith('_trans'):
                            # Map baseline to scenario-specific column
                            scenario_col = col_name.replace('_Baseline', f'_{scenario}')
                            if scenario_col in updated_row.index:
                                scenario_value = updated_row[scenario_col]
                                # Handle scalar values properly
                                try:
                                    if scenario_value is not None and not (isinstance(scenario_value, float) and np.isnan(scenario_value)):
                                        updated_row[col_name] = scenario_value
                                except (TypeError, ValueError):
                                    pass
                        
                        # Also update transformed variables
                        if '_Baseline_trans' in col_name:
                            scenario_col = col_name.replace('_Baseline_trans', f'_{scenario}_trans')
                            if scenario_col in updated_row.index:
                                scenario_value = updated_row[scenario_col]
                                # Handle scalar values properly
                                try:
                                    if scenario_value is not None and not (isinstance(scenario_value, float) and np.isnan(scenario_value)):
                                        updated_row[col_name] = scenario_value
                                except (TypeError, ValueError):
                                    pass
                
                # CRITICAL FIX: Set scenario-specific columns to NaN for consistency
                # These columns are not needed for regression since we use baseline columns
                for col in updated_row.index:
                    if isinstance(col, str):  # Fix linter error: ensure col is string
                        if ('_S1' in col or '_S3' in col or '_S4' in col) and col != f'_Baseline_{scenario}':
                            # Keep only the current scenario's values, set others to NaN
                            if not col.endswith(f'_{scenario}') and not col.endswith(f'_{scenario}_trans'):
                                updated_row[col] = np.nan
                
                # Update PD with GCorr forecast if available
                key = (country, scenario, date)
                if key in gcorr_updates:
                    updated_row['cdsiedf5_final'] = gcorr_updates[key]
                    updated_row['cdsiedf5'] = gcorr_updates[key]  # Also update non-adjusted for consistency
                
                forecast_updated.append(updated_row)
        
        # Combine all data
        all_records = []
        if not expanded_historical_df.empty:
            all_records.append(expanded_historical_df)
        
        if forecast_updated:
            forecast_df = pd.DataFrame(forecast_updated)
            forecast_df = forecast_df.reset_index(drop=True)
            all_records.append(forecast_df)
            
            print(f"Enhanced {len(forecast_df)} forecast records with GCorr annualized PD + scenario macro variables")
        
        if all_records:
            combined_df = pd.concat(all_records, ignore_index=True)
            combined_df = combined_df.sort_values(by=['cinc', 'scenario', 'yyyyqq'])
            
            print(f"Combined dataset: {len(expanded_historical_df)} historical + {len(forecast_updated) if forecast_updated else 0} forecast = {len(combined_df)} total records")
            
            return combined_df
        else:
            return expanded_historical_df if not expanded_historical_df.empty else historical_data
            
    except Exception as e:
        print(f"Error integrating transformed data: {e}")
        print("Falling back to GCorr PD only approach...")
        return create_gcorr_forecast_sample_pd_only(gcorr_data, historical_data)

def create_gcorr_forecast_sample_pd_only(gcorr_data, historical_data):
    """Fallback function - creates forecast using only GCorr annualized PD data (repeats last historical macro variables)"""
    
    if gcorr_data is None:
        return pd.DataFrame()
    
    # Get available countries from historical data
    historical_countries = set(historical_data['cinc'].unique())
    
    # Create expanded historical data for each scenario
    expanded_historical = []
    scenarios = ['S1', 'S3', 'S4', 'baseline']
    
    for country in historical_countries:
        country_hist = historical_data[historical_data['cinc'] == country].copy()
        if country_hist.empty:
            continue
            
        for scenario in scenarios:
            country_hist_scenario = country_hist.copy()
            country_hist_scenario['scenario'] = scenario
            country_hist_scenario['is_forecast'] = False
            expanded_historical.append(country_hist_scenario)
    
    expanded_historical_df = pd.concat(expanded_historical, ignore_index=True) if expanded_historical else pd.DataFrame()
    
    # Create forecast records using annualized PD directly
    forecast_records = []
    
    for _, row in gcorr_data.iterrows():
        country_iso = row['country_iso']
        scenario = row['scenario']
        date = row['date']
        annual_pd = row['annual_pd']
        
        if country_iso not in historical_countries:
            continue
        
        if pd.isna(annual_pd) or annual_pd <= 0:
            continue
        
        country_hist = historical_data[historical_data['cinc'] == country_iso].copy()
        if country_hist.empty:
            continue
        
        country_hist_sorted = country_hist.sort_values('yyyyqq')
        last_hist_record = country_hist_sorted.iloc[-1]
        
        # Create forecast record
        forecast_record = last_hist_record.copy()
        forecast_record['yyyyqq'] = date
        
        # Use annualized PD directly
        forecast_record['cdsiedf5_final'] = annual_pd
        forecast_record['cdsiedf5'] = annual_pd
        forecast_record['scenario'] = scenario
        forecast_record['is_forecast'] = True
        
        forecast_records.append(forecast_record)
    
    # Combine data
    all_records = []
    if not expanded_historical_df.empty:
        all_records.append(expanded_historical_df)
    
    if forecast_records:
        forecast_df = pd.DataFrame(forecast_records)
        all_records.append(forecast_df)
        # Created forecast records using GCorr annualized PD (repeating macro variables)
    
    if all_records:
        combined_df = pd.concat(all_records, ignore_index=True)
        combined_df = combined_df.sort_values(['cinc', 'scenario', 'yyyyqq'])
        return combined_df
    else:
        return expanded_historical_df if not expanded_historical_df.empty else historical_data



def calculate_dlnPD(data):
    """Calculate dlnPD (log change in PD) for combined historical and forecast data
    
    This function properly handles scenario dimensions to ensure continuity
    between historical and forecast data for each scenario trajectory.
    """
    data = data.copy()
    
    # Create PD column if it doesn't exist (use seasonality adjusted)
    if 'PD' not in data.columns:
        data['PD'] = data['cdsiedf5_final']
    
    # Calculate log PD
    data['lnPD'] = np.log(data['PD'])
    
    # If scenarios are present, group by both country and scenario
    # Otherwise, group by country only (historical data without scenarios)
    if 'scenario' in data.columns:
        # Sort by country, scenario, and date for proper ordering
        data = data.sort_values(['cinc', 'scenario', 'yyyyqq'])
        
        # Calculate dlnPD as the change in log PD within each country-scenario trajectory
        data['dlnPD'] = data.groupby(['cinc', 'scenario'])['lnPD'].diff()
        
        # Create lagged log PD for mean-reverting term within each trajectory
        data['lnPD_lag'] = data.groupby(['cinc', 'scenario'])['lnPD'].shift(1)
        
        print("Calculated dlnPD with scenario grouping for proper historical-forecast continuity")
    else:
        # Historical data only - group by country
        data = data.sort_values(['cinc', 'yyyyqq'])
        
        # Calculate dlnPD as the change in log PD
        data['dlnPD'] = data.groupby('cinc')['lnPD'].diff()
        
        # Create lagged log PD for mean-reverting term
        data['lnPD_lag'] = data.groupby('cinc')['lnPD'].shift(1)
        
        print("Calculated dlnPD with country grouping (historical data only)")
    
    return data

def check_sign_constraint(var, coef, sign_type):
    """Check if coefficient sign matches the required constraint"""
    if sign_type == 'flexible':
        return True  # Allow any sign
    elif sign_type == 'positive':
        return coef > 0
    elif sign_type == 'negative':
        return coef < 0
    return True  # Default to allowing any sign if constraint type is unknown

def extract_macro_variables_info(model_vars, coefficients, pvalues, stderr):
    """
    Extract macro variable information into separate columns.
    Returns dictionary with MV1-MV4 info, lags, coefficients, and p-values.
    """
    # Filter out fixed variables (mean_reverting, dlnPD_lag1) - no const for intercept=0
    fixed_vars = {meanrev_col, lag_col}
    macro_vars = [var for var in model_vars if var not in fixed_vars]
    
    # Initialize result dictionary
    result = {}
    
    # Extract up to 4 macro variables
    for i in range(4):
        mv_key = f'MV{i+1}'
        lag_key = f'MV{i+1}_lag'
        coef_key = f'MV{i+1}_coefficient'
        pval_key = f'MV{i+1}_pvalue'
        stderr_key = f'MV{i+1}_stderr'
        
        if i < len(macro_vars):
            var = macro_vars[i]
            
            # Extract base variable name and lag
            if '_lag' in var:
                base_var = var.rsplit('_lag', 1)[0]
                lag_num = var.split('_lag')[-1] if '_lag' in var else '0'
            else:
                base_var = var
                lag_num = '0'
            
            # Clean up variable name for display
            display_name = base_var.replace('_Baseline_trans', '')
            
            result[mv_key] = display_name
            result[lag_key] = int(lag_num) if lag_num.isdigit() else 0
            result[coef_key] = coefficients.get(var, np.nan)
            result[pval_key] = pvalues.get(var, np.nan)
            result[stderr_key] = stderr.get(var, np.nan)
        else:
            # Fill empty slots with NaN
            result[mv_key] = None
            result[lag_key] = None
            result[coef_key] = np.nan
            result[pval_key] = np.nan
            result[stderr_key] = np.nan
    
    # Add fixed variable information
    result['mean_reverting_coefficient'] = coefficients.get(meanrev_col, np.nan)
    result['mean_reverting_pvalue'] = pvalues.get(meanrev_col, np.nan)
    result['mean_reverting_stderr'] = stderr.get(meanrev_col, np.nan)
    
    if lag_col in model_vars:
        result['lag_coefficient'] = coefficients.get(lag_col, np.nan)
        result['lag_pvalue'] = pvalues.get(lag_col, np.nan)
        result['lag_stderr'] = stderr.get(lag_col, np.nan)
        result['includes_lag'] = True
    else:
        result['lag_coefficient'] = np.nan
        result['lag_pvalue'] = np.nan
        result['lag_stderr'] = np.nan
        result['includes_lag'] = False
    
    # No constant term for intercept=0 models
    result['constant_coefficient'] = 0.0
    result['constant_pvalue'] = np.nan
    result['constant_stderr'] = np.nan
    
    return result

# --- OVS PROCESS PER COUNTRY ---
def process_country(args):
    country, df_c, macro_vars = args
    results = []
    all_combinations = []  # Track all combinations and their status
    print(f"Processing country: {country}")
    
    # Use the filtered macro variables passed in
    available_vars = macro_vars
    
    # Pre-compute all lagged variables once
    print(f"  Pre-computing lagged variables...")
    mv_lag_options = {}
    for mv in available_vars:
        base = mv.replace('_Baseline_trans', '')
        mv_lag_options[base] = []
        for lag in range(0, max_lag+1):
            if lag == 0:
                var_name = f"{base}_Baseline_trans"
            else:
                var_name = f"{base}_Baseline_trans_lag{lag}"
                # Create lagged column if not present
                if var_name not in df_c.columns:
                    df_c[var_name] = df_c[f"{base}_Baseline_trans"].shift(lag)
            mv_lag_options[base].append(var_name)
    
    # Generate all combinations of macro variables with different lag options
    base_vars = list(mv_lag_options.keys())
    all_mv_combos = []
    
    # Generate combinations of 1 to max_k base variables
    for k in range(1, max_k+1):
        for base_combo in combinations(base_vars, k):
            # For each combination of base variables, generate all lag combinations
            lag_combinations = product(*[mv_lag_options[base] for base in base_combo])
            for lag_combo in lag_combinations:
                all_mv_combos.append(lag_combo)
    
    print(f"  Total model combinations: {len(all_mv_combos) * 2:,}")  # *2 for with/without dlnPD lag
    fixed_vars = [meanrev_col]
    
    # Progress tracking
    total_combos = len(all_mv_combos) * 2
    processed_count = 0
    
    # Pre-identify all unique macro variables for validation
    all_macro_vars = []
    for combo in all_mv_combos:
        all_macro_vars.extend(combo)
    unique_macro_vars = list(set(all_macro_vars))
    
    # Process all macro variable combinations for this country
    for mv_combo in all_mv_combos:
        # Try model with and without dlnPD lag term
        for include_lag in [True, False]:
            processed_count += 1
            
            # Print progress every 1%
            progress_interval = max(1, total_combos // 100)  # Changed from //10 to //100 for 1%
            if processed_count % progress_interval == 0 or processed_count == total_combos:
                progress_pct = (processed_count / total_combos) * 100
                print(f"    {country} Progress: {processed_count:,}/{total_combos:,} ({progress_pct:.1f}%)")
            
            X_cols = list(mv_combo) + fixed_vars
            if include_lag:
                X_cols.append(lag_col)
            
            # Check: sufficient observations
            df_model = df_c.dropna(subset=X_cols + [depvar])
            min_obs_required = max(30, 5 * len(X_cols))
            if len(df_model) < min_obs_required:
                continue
            
            # Early check: multicollinearity (only for combinations with >1 macro var)
            max_corr_val = 0.0
            if len(mv_combo) > 1:
                try:
                    corr_subset = df_model[list(mv_combo)].corr().abs()
                    upper = corr_subset.where(np.triu(np.ones(corr_subset.shape), k=1).astype(bool))
                    max_corr_val = upper.max().max()
                    if max_corr_val > max_corr:
                        continue  # Skip high correlation combinations
                except:
                    continue  # Skip if correlation calculation fails
            
            # Prepare data for regression
            X = df_model[X_cols]
            y = df_model[depvar]
            
            # Early check: ensure no constant columns (causes singular matrix)
            if (X.std() == 0).any():
                continue
            
            # Create weights: 2x weight for GCorr forecast data, 1x for historical data
            # if INCLUDE_GCORR_FORECAST and 'is_forecast' in df_model.columns:
            #     weights = df_model['is_forecast'].map({True: 2.0, False: 1.0}).fillna(1.0)
            # else:
            weights = None
            
            # X = sm.add_constant(X)  # Force intercept=0
            
            # Model estimation with error handling
            try:
                if weights is not None:
                    model = sm.WLS(y, X, weights=weights).fit()  # Weighted Least Squares
                else:
                    model = sm.OLS(y, X).fit()  # Regular OLS
            except:
                continue  # Skip failed estimations
            
            # Fast coefficient bounds check
            mean_rev_coef = model.params.get(meanrev_col, np.nan)
            if not (0 <= mean_rev_coef <= 1):
                continue
            
            if include_lag:
                lag_coef = model.params.get(lag_col, np.nan)
                if not (0 <= lag_coef <= 1):
                    continue
            
            # Fast p-value check
            pvals = model.pvalues.drop([meanrev_col] + ([lag_col] if include_lag else []), errors='ignore')
            if (pvals > max_pval).any():
                continue
            
            # Fast sign constraint check
            sign_violation = False
            for var in mv_combo:
                base = var.replace('_Baseline_trans', '').replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').replace('_lag4', '')
                sign_type = macro_var_sign_map.get(base)
                coef = model.params.get(var, np.nan)
                if not check_sign_constraint(base, coef, sign_type):
                    sign_violation = True
                    break
            
            if sign_violation:
                continue
            
            # If we reach here, the model is valid - extract information
            mv_info = extract_macro_variables_info(X_cols, model.params.to_dict(), 
                                                 model.pvalues.to_dict(), model.bse.to_dict())
            
            # Calculate adjusted R-squared manually for no-intercept models
            n = len(df_model)
            k = X.shape[1]  # Number of predictors (no intercept)
            r2 = model.rsquared
            
            # For no-intercept models: adj_r2 = 1 - (1 - r2) * n / (n - k)
            adj_r2_corrected = 1 - (1 - r2) * n / (n - k) if (n - k) > 0 else r2
            
            # Create result with improved format
            result = {
                'country': country,
                'n_obs': len(df_model),
                'r2': model.rsquared,
                'adj_r2': adj_r2_corrected,  # Use corrected calculation
                'AIC': model.aic,
                'BIC': model.bic,
                'MAE': np.mean(np.abs(model.resid)),
                'MSE': np.mean(model.resid**2),
                'RMSE': np.sqrt(np.mean(model.resid**2)),
                'MAPE': np.mean(np.abs(model.resid / y)),
                'max_correlation': max_corr_val
            }
            
            # Add the macro variables information
            result.update(mv_info)
            results.append(result)

    # Country-level outputs and plots removed - only return results for consolidated processing
    if not results:
        print(f"  WARNING: No valid models found for country {country}")
    
    return results, []  # Return empty list for combinations since we simplified tracking

def main():
    try:
        # Load historical data
        print("Loading historical data...")
        data = pd.read_csv(DATA_PATH)
        data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
        
        # TEMPORARY FILTER: Exclude Canada PD data before 2009 Q1
        canada_filter_date = pd.Timestamp('2009-01-01')  # 2009 Q1
        
        # Count records before filtering
        canada_before = len(data[(data['cinc'] == 'CAN') & (data['yyyyqq'] < canada_filter_date)])
        
        # Apply filter: Keep all non-Canada data + Canada data from 2009 Q1 onwards
        data = data[
            (data['cinc'] != 'CAN') |  # Keep all non-Canada data
            ((data['cinc'] == 'CAN') & (data['yyyyqq'] >= canada_filter_date))  # Keep Canada data from 2009 Q1+
        ].copy()
        
        print(f"  Filtered out {canada_before} Canada records before 2009 Q1")
        
        historical_data = data[data['yyyyqq'] < pd.Timestamp('2025-07-01')].copy()
        
        # Load GCorr forecast data if enabled
        if INCLUDE_GCORR_FORECAST:
            print(f"GCorr forecast option enabled - including {GCORR_FORECAST_QUARTERS} quarters")
            gcorr_data = load_gcorr_forecast_data()
            forecast_data = create_gcorr_forecast_sample(gcorr_data, historical_data)
            
            if not forecast_data.empty:
                # Use only forecast_data which already contains historical + forecast for all scenarios
                data = forecast_data
                data = data.sort_values(by=['cinc', 'scenario', 'yyyyqq'])
                print(f"Using scenario-based data: {len(data)} total records across all scenarios")
            else:
                data = historical_data
                print("No forecast data available, using historical data only")
        else:
            data = historical_data
            print("GCorr forecast option disabled, using historical data only")
        
        # Identify Baseline macro variable columns 
        baseline_mv_cols = [col for col in data.columns if col.endswith('_Baseline_trans')]
        
        # Exclude specific variables from OVS selection
        excluded_vars = [
            'Monetary Policy Rate_Baseline_trans',
            'Government 10Y Bond Rate_Baseline_trans',
            # 'FX_Baseline_trans',
            # 'Term Spread_Baseline_trans'
        ]
        
        macro_vars = [col for col in baseline_mv_cols if col not in [depvar, lag_col, meanrev_col] + excluded_vars]
        
        print(f"Total macro variables for selection: {len(macro_vars)}")
        
        # Prepare data
        data = calculate_dlnPD(data)
        
        # Apply winsorization to dlnPD
        data, winsorization_records = winsorize_dlnPD(data, STD_MULTIPLIER)
        
        # Handle scenario grouping if present
        if 'scenario' in data.columns:
            data = data.sort_values([country_col, 'scenario', 'yyyyqq'])
            # Create lag from winsorized dlnPD so both are consistent - group by country and scenario
            data[lag_col] = data.groupby([country_col, 'scenario'])[depvar].shift(1)
        else:
            data = data.sort_values([country_col, 'yyyyqq'])
            data[lag_col] = data.groupby(country_col)[depvar].shift(1)
        
        # Compute historical average PD (TTC PD) for each country - use only historical data for TTC
        if INCLUDE_GCORR_FORECAST:
            # For TTC calculation, use only historical data (where is_forecast is False or not present)
            if 'is_forecast' in data.columns:
                historical_mask = (data['is_forecast'] == False) | data['is_forecast'].isna()
            else:
                historical_mask = data['yyyyqq'] < pd.Timestamp('2025-07-01')
            historical_ttc = data[historical_mask].groupby(country_col)['PD'].mean()
            # Use merge to avoid deprecated replace downcasting behavior
            ttc_mapping = historical_ttc.reset_index()
            ttc_mapping.columns = [country_col, ttc_col]
            data = data.merge(ttc_mapping[[country_col, ttc_col]], on=country_col, how='left')
        else:
            data[ttc_col] = data.groupby(country_col)['PD'].transform('mean')
        data[meanrev_col] = np.log(data[ttc_col]) - data['lnPD_lag']
        
        # Remove duplicated historical data when GCorr forecast is included
        if INCLUDE_GCORR_FORECAST and 'scenario' in data.columns:
            # Keep only one copy of historical data (from baseline scenario) and all forecast data
            historical_data_baseline = data[(data['scenario'] == 'baseline') & 
                                          ((data['is_forecast'] == False) | data['is_forecast'].isna())]
            forecast_data_all = data[data['is_forecast'] == True]
            
            # Combine baseline historical + all forecast scenarios
            data = pd.concat([historical_data_baseline, forecast_data_all], ignore_index=True)
            if 'scenario' in data.columns:
                data = data.sort_values([country_col, 'scenario', 'yyyyqq'])
            else:
                data = data.sort_values([country_col, 'yyyyqq'])
            
            print(f"Removed duplicated historical data: {len(historical_data_baseline)} historical + {len(forecast_data_all)} forecast = {len(data)} total records")
        
        # Drop rows with missing values
        data = data.dropna(subset=[depvar, lag_col, meanrev_col] + macro_vars, how='all')
        
        # Save prepared data if requested
        if SAVE_PREPARED_DATA:
            gcorr_suffix = f'_gcorr{GCORR_FORECAST_QUARTERS}q' if INCLUDE_GCORR_FORECAST else ''
            smoothed_suffix = '_smoothed' if (INCLUDE_GCORR_FORECAST and GCORR_USE_SMOOTHED_PD) else ''
            prepared_data_file = OUTPUT_DIR / f'prepared_data{gcorr_suffix}{smoothed_suffix}.csv'
            data.to_csv(prepared_data_file, index=False)
            print(f"\nPrepared data saved to: {prepared_data_file}")
            print(f"Data shape: {data.shape}")
            print(f"Countries: {len(data[country_col].unique())}")
            print(f"Date range: {data['yyyyqq'].min().strftime('%Y-%m')} to {data['yyyyqq'].max().strftime('%Y-%m')}")
            
            if 'scenario' in data.columns:
                scenarios_unique = data['scenario'].dropna().unique()
                print(f"Scenarios: {sorted([str(s) for s in scenarios_unique])}")
            
            if 'is_forecast' in data.columns:
                forecast_counts = data['is_forecast'].value_counts()
                print("Records by type:")
                for is_forecast, count in forecast_counts.items():
                    data_type = 'Forecast' if is_forecast else 'Historical'
                    print(f"  {data_type}: {count:,}")
        
        # Exit if only data preparation requested
        if DATA_PREPARATION_ONLY:
            print(f"\nDATA PREPARATION COMPLETE - Exiting before variable selection")
            print(f"=" * 60)
            print(f"FILES CREATED:")
            if SAVE_PREPARED_DATA:
                print(f"1. {prepared_data_file.name} - Complete prepared dataset")
                print(f"   Contains: Historical + GCorr forecast data with calculated dlnPD, lags, mean-reverting terms")
            if INCLUDE_GCORR_FORECAST:
                gcorr_suffix = f'_gcorr{GCORR_FORECAST_QUARTERS}q' if INCLUDE_GCORR_FORECAST else ''
                report_file = OUTPUT_DIR / f'data_validation_report{gcorr_suffix}.txt'
                print(f"2. {report_file.name} - Detailed validation report")
                print(f"   Contains: Continuity checks for all countries and scenarios")
            print(f"=" * 60)
            print(f"NEXT STEPS:")
            print(f"1. Review the prepared data file to check data structure")
            print(f"2. Check the validation report for any continuity issues")
            print(f"3. Set DATA_PREPARATION_ONLY = False to run variable selection")
            print(f"4. Adjust GCORR_FORECAST_QUARTERS if needed (currently {GCORR_FORECAST_QUARTERS})")
            return
        
        # Run OVS process
        results = []
        countries = data[country_col].unique()
        
        # Filter to test country if in test mode
        if TEST_MODE:
            if TEST_COUNTRY in countries:
                countries = [TEST_COUNTRY]
                print(f"\n*** TEST MODE: Processing only {TEST_COUNTRY} ***")
            else:
                print(f"\n*** ERROR: Test country {TEST_COUNTRY} not found in data ***")
                print(f"Available countries: {sorted(countries)}")
                return
        
        # Check if we have scenario data
        has_scenarios = 'scenario' in data.columns
        if has_scenarios:
            print(f"\nDetected scenario data - will pool all scenarios for each country's model fitting")
        else:
            print(f"\nNo scenario data detected - using historical data only")
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            country_args = [(country, data[data[country_col] == country].copy(), macro_vars) for country in countries]
            country_outputs = list(executor.map(process_country, country_args))
        
        # Combine results
        all_results = []
        for country_results, _ in country_outputs:  # Ignore empty combinations list
            all_results.extend(country_results)
        
        # Save results
        if all_results:
            # Create simple winsorization summary file
            gcorr_suffix = f'_gcorr{GCORR_FORECAST_QUARTERS}q' if INCLUDE_GCORR_FORECAST else ''
            smoothed_suffix = '_smoothed' if (INCLUDE_GCORR_FORECAST and GCORR_USE_SMOOTHED_PD) else ''
            if winsorization_records:
                winsorization_df = pd.DataFrame(winsorization_records)
                winsorization_df.to_csv(OUTPUT_DIR / f'winsorization_summary{gcorr_suffix}{smoothed_suffix}.csv', index=False)
            
            # Save consolidated results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(OUTPUT_DIR / f'ovs_results{gcorr_suffix}{smoothed_suffix}.csv', index=False)
            
            # Calculate variable usage from MV columns
            mv_usage = {}
            country_model_counts = {}
            for result in all_results:
                # Track variable usage
                for i in range(1, 5):  # MV1 to MV4
                    mv_col = f'MV{i}'
                    if result.get(mv_col) is not None:
                        var_name = result[mv_col]
                        mv_usage[var_name] = mv_usage.get(var_name, 0) + 1
                
                # Track models per country
                country = result['country']
                country_model_counts[country] = country_model_counts.get(country, 0) + 1
            
            # Create summary
            summary = {
                'total_countries': len(countries),
                'countries_with_models': len(set(r['country'] for r in all_results)),
                'total_models': len(all_results),
                'avg_models_per_country': len(all_results) / len(set(r['country'] for r in all_results)),
                'variable_usage': mv_usage,
                'country_model_counts': country_model_counts,
                'winsorization_summary': {
                    'enabled': APPLY_WINSORIZATION,
                    'std_multiplier': STD_MULTIPLIER,
                    'records_winsorized': len(winsorization_records) if winsorization_records else 0
                },
                'gcorr_forecast_summary': {
                    'enabled': INCLUDE_GCORR_FORECAST,
                    'forecast_quarters': GCORR_FORECAST_QUARTERS if INCLUDE_GCORR_FORECAST else 0,
                    'gcorr_data_path': str(GCORR_FORECAST_DIR) if INCLUDE_GCORR_FORECAST else None,
                    'historical_records': len(historical_data),
                    'total_records': len(data),
                    'forecast_records': len(data) - len(historical_data) if INCLUDE_GCORR_FORECAST else 0,
                    'quarterly_to_annual_conversion': 'Applied' if INCLUDE_GCORR_FORECAST else 'N/A',
                    'conversion_formula': 'PD_annual = 1 - (1 - PD_quarterly)^4' if INCLUDE_GCORR_FORECAST else 'N/A'
                }
            }
            
            with open(OUTPUT_DIR / f'ovs_summary{gcorr_suffix}{smoothed_suffix}.json', 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(v) for v in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    else:
                        return obj
                
                summary_serializable = convert_numpy_types(summary)
                json.dump(summary_serializable, f, indent=2)
            
            print(f"\nOVS results saved to: {OUTPUT_DIR}")
            print(f"Total models found: {len(all_results):,}")
            print(f"Countries with models: {summary['countries_with_models']} out of {summary['total_countries']}")
            
            # Print GCorr forecast summary
            gcorr_summary = summary['gcorr_forecast_summary']
            if gcorr_summary['enabled']:
                print(f"\nGCorr Forecast Summary:")
                print(f"  Forecast quarters included: {gcorr_summary['forecast_quarters']}")
                print(f"  Total records: {gcorr_summary['total_records']:,} (historical + forecast)")
                print(f"  Quarterly to annual conversion: {gcorr_summary['quarterly_to_annual_conversion']}")
            
            # Print top 10 most used variables
            print("\nTop 10 Most Used Variables:")
            if mv_usage:
                sorted_vars = sorted(mv_usage.items(), key=lambda x: x[1], reverse=True)[:10]
                for var, count in sorted_vars:
                    print(f"  {var}: {count} models")
            else:
                print("  No macro variables found in results")
        else:
            print("\nNo valid models found")
        
    except Exception as e:
        print(f"Error in OVS process: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pass  # No plotting cleanup needed

if __name__ == "__main__":
    main() 