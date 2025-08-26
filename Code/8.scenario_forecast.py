import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
import re
import shutil
warnings.filterwarnings('ignore')

# Output directories
output_dir = Path('Output/8.scenario_forecast')
output_dir.mkdir(parents=True, exist_ok=True)
plots_dir = output_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

# Create centralized comparison directories for different lag options
comparison_plots_dir_ignore_lags = output_dir / 'charts_ignore_lags'
comparison_plots_dir_ignore_lags.mkdir(parents=True, exist_ok=True)

comparison_plots_dir_has_no_lags = output_dir / 'charts_has_no_lags'
comparison_plots_dir_has_no_lags.mkdir(parents=True, exist_ok=True)

# Configuration
DATA_PATH = 'Output/4.transformation/transformed_data.csv'
OVS_RESULTS_DIR = Path('Output/6.ovs_variable_selection')
FORECAST_START = '2025-07-01'  # 2025Q3
GCORR_FORECAST_QUARTERS = 40  # Number of GCorr forecast quarters (20 or 40)

# New configuration for multiple OVS sources and lag options
OVS_SOURCES = {
    # 'original': 'Output/6.ovs_variable_selection/ovs_results.csv',
    'advanced': 'Output/7.filtered_ovs_results/final_top_models_advanced.csv',
    # 'original_gcorr': f'Output/6.ovs_variable_selection/ovs_results_gcorr{GCORR_FORECAST_QUARTERS}q.csv',
    f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q': f'Output/7.filtered_ovs_results/final_top_models_advanced_gcorr{GCORR_FORECAST_QUARTERS}q.csv',
    # 'original_gcorr_smoothed': f'Output/6.ovs_variable_selection/ovs_results_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed.csv',
    f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed': f'Output/7.filtered_ovs_results/final_top_models_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed.csv'
}

LAG_OPTIONS = {
    # 'ignore_lags': 'ignore_mv_lags',      # Current implementation - ignore MV lags
    'has_no_lags': 'filter_no_lags',  # New option - filter for models with has_no_lags == TRUE
    # 'with_lags': 'use_mv_lags'        # New option - use MV lags from OVS model
}

def get_top_model_for_country(country, ovs_source, lag_option='ignore_lags'):
    """
    Helper function to get the top model for a country from final top models files.
    Returns the model row or None if not found.
    """
    try:
        ovs_file_path = OVS_SOURCES[ovs_source]
        ovs_data = pd.read_csv(ovs_file_path)
        
        # For final top models files, simply get the model for the country
        # These files already contain only the selected top model per country
        country_models = ovs_data[ovs_data['country'] == country]
        
        # Apply has_no_lags filter if requested and column exists
        if lag_option == 'has_no_lags' and 'has_no_lags' in ovs_data.columns:
            country_models = country_models[country_models['has_no_lags'] == True]
            
        if not country_models.empty:
            # Return the first (and should be only) model for this country
            return country_models.iloc[0]
        
        return None
        
    except Exception as e:
        print(f"  Warning: Could not load model for {country} from {ovs_source}: {str(e)}")
        return None

def create_lagged_variables(data, model_info):
    """Create lagged macro variables if lag_option is 'with_lags'"""
    lag_option = model_info.get('lag_option', 'no_lags')
    
    if lag_option == 'no_lags':
        return data
    
    # Create a copy of the data to avoid modifying the original
    data_with_lags = data.copy()
    
    # Get the model row to determine lag information
    country = data['cinc'].iloc[0]
    ovs_source = model_info.get('ovs_source', 'filtered')
    model_row = get_top_model_for_country(country, ovs_source)
    
    if model_row is None:
        return data_with_lags
    
    # Create lagged variables for each macro variable
    for i in range(1, 5):  # MV1 to MV4
        mv_col = f'MV{i}'
        lag_col = f'MV{i}_lag'
        
        if (mv_col in model_row and pd.notna(model_row[mv_col]) and 
            lag_col in model_row and pd.notna(model_row[lag_col]) and 
            model_row[lag_col] > 0):
            
            base_mv_name = model_row[mv_col]
            lag_periods = int(model_row[lag_col])
            
            # Create lagged versions for all scenarios
            scenarios = ['Baseline', 'S1', 'S3', 'S4']
            for scenario in scenarios:
                original_col = f"{base_mv_name}_{scenario}_trans"
                lagged_col = f"{base_mv_name}_lag{lag_periods}_{scenario}_trans"
                
                if original_col in data_with_lags.columns:
                    # Create lagged variable by shifting
                    data_with_lags[lagged_col] = data_with_lags[original_col].shift(lag_periods)
    
    return data_with_lags

def get_macro_variables_for_plotting(country, ovs_source, lag_option):
    """
    Helper function to get macro variable names for plotting.
    Returns list of variable names to plot (with or without lag suffixes).
    """
    model_row = get_top_model_for_country(country, ovs_source, lag_option)
    
    if model_row is None:
        return []
    
    macro_vars = []
    for i in range(1, 5):  # MV1 to MV4
        mv_col = f'MV{i}'
        lag_col = f'MV{i}_lag'
        
        if mv_col in model_row and pd.notna(model_row[mv_col]) and model_row[mv_col].strip():
            base_mv_name = model_row[mv_col]
            
            # Determine which variable name to use for plotting
            if lag_option == 'with_lags' and lag_col in model_row and pd.notna(model_row[lag_col]) and model_row[lag_col] > 0:
                # Use lagged variable name for plotting
                lag_periods = int(model_row[lag_col])
                plot_mv_name = f"{base_mv_name}_lag{lag_periods}"
            else:
                # Use original variable name
                plot_mv_name = base_mv_name
            
            macro_vars.append(plot_mv_name)
    
    return macro_vars

def get_model_description_for_plotting(country, ovs_source, lag_option):
    """
    Helper function to create model description string for plot titles.
    Returns formatted string with variable names, lags, and coefficients.
    """
    model_row = get_top_model_for_country(country, ovs_source, lag_option)
    
    if model_row is None:
        return 'Model variables not available'
    
    clean_mv_names = []
    
    # Skip constant term - don't add it to model description
    # if 'constant_coefficient' in model_row and pd.notna(model_row['constant_coefficient']):
    #     clean_mv_names.append(f"Const({model_row['constant_coefficient']:.2f})")
    
    # Add dlnPD lag if present
    if 'includes_lag' in model_row and model_row['includes_lag'] and pd.notna(model_row['lag_coefficient']):
        clean_mv_names.append(f"dlnPD_lag1({model_row['lag_coefficient']:.2f})")
    
    # Add mean reverting if present
    if 'mean_reverting_coefficient' in model_row and pd.notna(model_row['mean_reverting_coefficient']):
        clean_mv_names.append(f"MeanRev({model_row['mean_reverting_coefficient']:.2f})")
    
    # Add macro variables with lags and coefficients
    for i in range(1, 5):  # MV1 to MV4
        mv_col = f'MV{i}'
        lag_col = f'MV{i}_lag'
        coeff_col = f'MV{i}_coefficient'
        
        if (mv_col in model_row and pd.notna(model_row[mv_col]) and 
            coeff_col in model_row and pd.notna(model_row[coeff_col])):
            
            mv_name = model_row[mv_col]
            lag = model_row[lag_col] if lag_col in model_row and pd.notna(model_row[lag_col]) else 0
            coeff = model_row[coeff_col]
            
            # Show the lag information based on the lag_option
            if lag_option == 'with_lags' and lag > 0:
                clean_mv_names.append(f"{mv_name}_lag{int(lag)}({coeff:.2f})")
            else:
                clean_mv_names.append(f"{mv_name}({coeff:.2f})")
    
    return ' + '.join(clean_mv_names) if clean_mv_names else 'Model variables not available'

def load_ovs_results(ovs_source='advanced', lag_option='ignore_lags'):
    """Load OVS results and return top model for each country from final top models files
    
    Args:
        ovs_source: 'advanced', f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q', 
                   or f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed'
        lag_option: 'ignore_lags' (ignore MV lags) or 'with_lags' (use MV lags)
    """
    ovs_results = {}  # Initialize the results dictionary
    
    # Get the appropriate OVS file path
    ovs_file = Path(OVS_SOURCES[ovs_source])
    
    if not ovs_file.exists():
        raise FileNotFoundError(f"OVS results file not found: {ovs_file}")
    
    all_results = pd.read_csv(ovs_file)
    # Final top models files already contain only the selected top model per country
    top_models = all_results
    
    # Process each country to get the appropriate model based on lag_option
    countries = top_models['country'].unique()
    for country in countries:
        # Get the appropriate model for this country based on lag_option
        model = get_top_model_for_country(country, ovs_source, lag_option)
        if model is None:
            continue
            
        # Build model_vars and coefficients
        model_vars = []
        coefficients = []
        
        # Add intercept
        if 'constant_coefficient' in model.index and pd.notna(model['constant_coefficient']):
            model_vars.append('const')
            coefficients.append(model['constant_coefficient'])
        
        # Add dlnPD lags
        if 'includes_lag' in model.index and model['includes_lag'] and pd.notna(model['lag_coefficient']):
            model_vars.append('dlnPD_l1')  # The OVS uses lag1 for dlnPD
            coefficients.append(model['lag_coefficient'])
        
        # Add mean reverting term
        if 'mean_reverting_coefficient' in model.index and pd.notna(model['mean_reverting_coefficient']):
            model_vars.append('mean_reverting')
            coefficients.append(model['mean_reverting_coefficient'])
        
        # Add macro variables - handling lag option
        for i in range(1, 5):  # MV1-MV4
            mv_col = f'MV{i}'
            coeff_col = f'MV{i}_coefficient'
            lag_col = f'MV{i}_lag'
            
            if mv_col in model.index and pd.notna(model[mv_col]) and model[mv_col].strip():
                base_mv_name = model[mv_col].split('_lag')[0] if '_lag' in model[mv_col] else model[mv_col]
                
                if lag_option == 'with_lags' and lag_col in model.index and pd.notna(model[lag_col]) and model[lag_col] > 0:
                    # Use lagged macro variable
                    mv_name = f"{base_mv_name}_lag{int(model[lag_col])}_Baseline_trans"
                elif lag_option == 'ignore_lags':
                    # Force unshifted macro variable (ignore any lags from OVS)
                    mv_name = f"{base_mv_name}_Baseline_trans"
                else:  # has_no_lags
                    # Use variables as they were selected (should already be unshifted since has_no_lags==True)
                    # but double-check to ensure no lags
                    if lag_col in model.index and pd.notna(model[lag_col]) and model[lag_col] > 0:
                        print(f"Warning: has_no_lags model has lag > 0 for {base_mv_name}, using unshifted version")
                    mv_name = f"{base_mv_name}_Baseline_trans"
                
                model_vars.append(mv_name)
                coefficients.append(model[coeff_col])
            
            ovs_results[country] = {
                'model_vars': model_vars,
                'coefficients': coefficients,
                'adj_r2': model['adj_r2'],
                'rank': 1,  # Final top models files contain only one model per country
                'has_no_lags': model.get('has_no_lags', False),
                'ovs_source': ovs_source,
                'lag_option': lag_option
            }
        
        # Model details available in output files
    
    print(f"Loaded OVS results for {len(ovs_results)} countries (source: {ovs_source}, lags: {lag_option})")
    return ovs_results

def identify_scenarios(data):
    """Identify available scenarios in the data"""
    scenario_patterns = set()
    
    # Check for scenario naming patterns using regex
    for col in data.columns:
        # Look for patterns like _Baseline_trans, _S1_trans, _S2_trans, etc.
        match = re.search(r'_([^_]+)_trans$', col)
        if match:
            scenario = match.group(1)
            # Include Baseline and S1-S9 patterns
            if scenario == 'Baseline' or re.match(r'^S[1-9]$', scenario):
                scenario_patterns.add(scenario)
    
    scenarios = sorted(list(scenario_patterns))
    print(f"Found scenarios: {scenarios}")
    return scenarios

def prepare_scenario_data(data, country, scenarios, model_info=None):
    """Prepare data for each scenario"""
    scenario_data = {}
    
    # Filter for specific country
    country_data = data[data['cinc'] == country].copy().sort_values('yyyyqq')
    
    # Create lagged variables if needed
    if model_info and model_info.get('lag_option') == 'with_lags':
        country_data = create_lagged_variables(country_data, model_info)
    
    for scenario in scenarios:
        scenario_df = country_data.copy()
        scenario_data[scenario] = scenario_df
    
    return scenario_data

def map_model_vars_to_scenario(model_vars, scenario, lag_option='ignore_lags'):
    """
    Map model variable names from Baseline to scenario-specific names.
    
    Args:
        model_vars: List of model variable names
        scenario: Target scenario name
        lag_option: How to handle lags ('no_lags' vs 'has_no_lags')
    
    Note: 
    - 'ignore_lags': Always use unshifted (lag 0) macro variables for forecasting
    - 'has_no_lags': Use variables as-is (these models were selected without lags)
    """
    if scenario == 'Baseline':
        return model_vars
    
    mapped_vars = []
    for var in model_vars:
        if '_Baseline_trans' in var:
            if lag_option == 'ignore_lags':
                # Force unshifted variables (ignore any lags from OVS)
                mapped_var = var.replace('_Baseline_trans', f'_{scenario}_trans')
            else:  # has_no_lags
                # Use variables as-is - these models were selected without lags
                mapped_var = var.replace('_Baseline_trans', f'_{scenario}_trans')
            mapped_vars.append(mapped_var)
        else:
            # Keep non-baseline variables as is (dlnPD_lag1, mean_reverting, const)
            mapped_vars.append(var)
    
    return mapped_vars

def forecast_dlnpd(historical_data, forecast_data, model_info, scenario='Baseline'):
    """Forecast dlnPD using the OVS model"""
    model_vars = model_info['model_vars']
    coefficients = model_info['coefficients']
    lag_option = model_info.get('lag_option', 'no_lags')
    
    # Convert coefficients list to dictionary for easier access
    coeff_dict = dict(zip(model_vars, coefficients))
    
    # Map model variables to scenario-specific names
    scenario_model_vars = map_model_vars_to_scenario(model_vars, scenario, lag_option)
    
    # Combine historical and forecast data for continuous series
    combined_data = pd.concat([historical_data, forecast_data], ignore_index=True)
    combined_data = combined_data.sort_values('yyyyqq').reset_index(drop=True)
    
    # Identify forecast period
    forecast_start_dt = pd.to_datetime(FORECAST_START)
    is_forecast = combined_data['yyyyqq'] >= forecast_start_dt
    
    # Initialize forecast results
    forecast_results = []
    
    # Get the last historical dlnPD value for initialization
    hist_mask = combined_data['yyyyqq'] < forecast_start_dt
    if hist_mask.any():
        last_hist_dlnpd = combined_data.loc[hist_mask, 'dlnPD'].dropna().iloc[-1] if not combined_data.loc[hist_mask, 'dlnPD'].dropna().empty else 0
    else:
        last_hist_dlnpd = 0
    
    current_dlnpd = last_hist_dlnpd
    current_lnpd = None  # Track current log PD level
    
    # Initialize current log PD from last historical value
    if hist_mask.any():
        last_hist_pd = combined_data.loc[hist_mask, 'cdsiedf5'].dropna().iloc[-1] if not combined_data.loc[hist_mask, 'cdsiedf5'].dropna().empty else 0.001
        current_lnpd = np.log(last_hist_pd)
    else:
        current_lnpd = np.log(0.001)  # Default small value
    
    # Generate recursive forecasts
    for idx in combined_data.index:
        if is_forecast[idx]:
            # Prepare input variables
            X_values = {}
            
            # Update lagged dlnPD variables with recent predictions
            if 'dlnPD_l1' in model_vars:
                combined_data.loc[idx, 'dlnPD_l1'] = current_dlnpd
            
            # Update lnPD_lag with current log PD level
            if 'lnPD_lag' in combined_data.columns:
                combined_data.loc[idx, 'lnPD_lag'] = current_lnpd
            
            # Update mean-reverting term with recent PD level
            if 'mean_reverting' in model_vars:
                # Calculate TTC (through-the-cycle) PD - historical average
                hist_pd = combined_data.loc[hist_mask, 'cdsiedf5'].dropna()
                if not hist_pd.empty:
                    ttc_pd = hist_pd.mean()
                    ln_ttc_pd = np.log(ttc_pd)
                    # Mean-reverting term: ln(TTC_PD) - lnPD_lag (use existing column)
                    if 'lnPD_lag' in combined_data.columns:
                        lnpd_lag_val = combined_data.loc[idx, 'lnPD_lag']
                        mean_rev_val = ln_ttc_pd - lnpd_lag_val
                        combined_data.loc[idx, 'mean_reverting'] = mean_rev_val
                    else:
                        combined_data.loc[idx, 'mean_reverting'] = 0
                else:
                    combined_data.loc[idx, 'mean_reverting'] = 0
            
            # Extract model variables using scenario-specific names
            for orig_var, scenario_var in zip(model_vars, scenario_model_vars):
                if orig_var == 'const':
                    X_values[orig_var] = 1.0
                elif scenario_var in combined_data.columns:
                    val = combined_data.loc[idx, scenario_var]
                    X_values[orig_var] = val
                else:
                    # Fallback to original variable name if scenario-specific doesn't exist
                    if orig_var in combined_data.columns:
                        val = combined_data.loc[idx, orig_var]
                        X_values[orig_var] = val
                    else:
                        X_values[orig_var] = 0.0  # Default value for missing variables
            
            # Calculate predicted dlnPD using coefficient dictionary
            predicted_dlnpd = sum(coeff_dict.get(var, 0) * X_values.get(var, 0) for var in model_vars)
            
            # Calculate predicted PD: exp(current_lnPD + dlnPD)
            new_lnpd = current_lnpd + predicted_dlnpd
            predicted_pd = np.exp(new_lnpd)
            
            # Store results
            forecast_results.append({
                'yyyyqq': combined_data.loc[idx, 'yyyyqq'],
                'predicted_dlnPD': predicted_dlnpd,
                'predicted_PD': predicted_pd
            })
            
            # Update current values for next iteration
            current_dlnpd = predicted_dlnpd
            current_lnpd = new_lnpd  # Use the new log PD level
    
    return pd.DataFrame(forecast_results)

def forecast_historical_1step(historical_data, model_info):
    """Generate 1-step ahead forecasts during historical period for model validation"""
    model_vars = model_info['model_vars']
    coefficients = model_info['coefficients']
    
    # Convert coefficients list to dictionary for easier access
    coeff_dict = dict(zip(model_vars, coefficients))
    
    # Sort data by date and filter to only PD data period
    hist_data = historical_data.copy().sort_values('yyyyqq').reset_index(drop=True)
    
    # Only forecast where PD data actually exists (not NaN)
    pd_mask = hist_data['cdsiedf5'].notna()
    hist_data = hist_data[pd_mask].reset_index(drop=True)
    
    if len(hist_data) < 2:
        print("    Insufficient PD data for historical forecasting")
        return pd.DataFrame()
    
    # Initialize results
    forecast_results = []
    
    # Start from the second observation (need lag for first forecast)
    for idx in range(1, len(hist_data)):
        try:
            # Prepare input variables for current time point
            X_values = {}
            
            # Extract model variables
            for var in model_vars:
                if var == 'const':
                    X_values[var] = 1.0
                elif var in hist_data.columns:
                    val = hist_data.loc[idx, var]
                    X_values[var] = val if not pd.isna(val) else 0.0
                else:
                    X_values[var] = 0.0
            
            # Calculate predicted dlnPD using coefficient dictionary
            predicted_dlnpd = sum(coeff_dict.get(var, 0) * X_values.get(var, 0) for var in model_vars)
            
            # Calculate predicted PD level correctly
            # Method 1: Use lnPD_lag if available
            if 'lnPD_lag' in hist_data.columns and not pd.isna(hist_data.loc[idx, 'lnPD_lag']):
                current_lnpd = hist_data.loc[idx, 'lnPD_lag']
                predicted_lnpd = current_lnpd + predicted_dlnpd
                predicted_pd = np.exp(predicted_lnpd)
            else:
                # Method 2: Use previous period's actual PD
                if idx > 0 and not pd.isna(hist_data.loc[idx-1, 'cdsiedf5']):
                    prev_pd = hist_data.loc[idx-1, 'cdsiedf5']
                    if prev_pd > 0:
                        predicted_pd = prev_pd * np.exp(predicted_dlnpd)
                    else:
                        predicted_pd = 0.001 * np.exp(predicted_dlnpd)  # Small default
                else:
                    # Fallback: use small default PD
                    predicted_pd = 0.001 * np.exp(predicted_dlnpd)
            
            # Get actual values for comparison
            actual_dlnpd = hist_data.loc[idx, 'dlnPD'] if 'dlnPD' in hist_data.columns else np.nan
            actual_pd = hist_data.loc[idx, 'cdsiedf5'] if 'cdsiedf5' in hist_data.columns else np.nan
            
            forecast_results.append({
                'yyyyqq': hist_data.loc[idx, 'yyyyqq'],
                'predicted_dlnPD': predicted_dlnpd,
                'predicted_PD': predicted_pd,
                'actual_dlnPD': actual_dlnpd,
                'actual_PD': actual_pd
            })
            
        except Exception as e:
            # Skip problematic observations
            continue
    
    result_df = pd.DataFrame(forecast_results)
    print(f"    Generated {len(result_df)} historical forecasts (PD period: {hist_data['yyyyqq'].min().strftime('%Y-%m')} to {hist_data['yyyyqq'].max().strftime('%Y-%m')})")
    
    return result_df

def plot_scenario_comparison(country, historical_data, scenario_forecasts, ovs_model, historical_forecast=None, 
                           plots_dir=None, combination_name=""):
    """Plot comparison of scenarios for a country with optional historical 1-step forecasts"""
    if plots_dir is None:
        plots_dir = Path('Output/8.scenario_forecast/plots')
    
    # Set larger font sizes for all plot elements
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    
    # Define cutoff date for display (before 2040)
    cutoff_date = pd.to_datetime('2040-01-01')
    
    # === PD LEVELS PLOT ===
    # Plot historical PD
    historical_data = historical_data.sort_values('yyyyqq')
    # Only plot where PD data exists and before cutoff
    hist_pd_data = historical_data[
        (historical_data['cdsiedf5'].notna()) & 
        (historical_data['yyyyqq'] < cutoff_date)
    ]
    if not hist_pd_data.empty:
        ax1.plot(hist_pd_data['yyyyqq'], hist_pd_data['cdsiedf5'] * 100, 
                 label='Historical PD (Actual)', color='black', linewidth=3)
        
        # Get the last historical PD point for connecting forecasts
        last_hist_date = hist_pd_data['yyyyqq'].iloc[-1]
        last_hist_pd = hist_pd_data['cdsiedf5'].iloc[-1] * 100
    else:
        last_hist_date = None
        last_hist_pd = None
    
    # Plot historical 1-step forecasts if available
    if historical_forecast is not None and not historical_forecast.empty:
        historical_forecast = historical_forecast.sort_values('yyyyqq')
        # Filter by cutoff date and validate predicted PD values
        historical_forecast_filtered = historical_forecast[
            historical_forecast['yyyyqq'] < cutoff_date
        ]
        valid_pred = historical_forecast_filtered['predicted_PD'].between(1e-8, 2.0)
        if valid_pred.sum() > 0:
            plot_data = historical_forecast_filtered[valid_pred]
            ax1.plot(plot_data['yyyyqq'], plot_data['predicted_PD'] * 100, 
                     label='1-Step Ahead Forecast (Historical)', color='gray', 
                     linestyle='-', alpha=0.7, linewidth=2)
        else:
            print(f"    Warning: No valid predicted PD values for {country} (range: {historical_forecast_filtered['predicted_PD'].min():.2e} to {historical_forecast_filtered['predicted_PD'].max():.2e})")
    
    # Plot forecast scenarios with connection to last historical point
    colors = ['blue', 'orange', 'red', 'green', 'purple']
    for i, (scenario, forecast_df) in enumerate(scenario_forecasts.items()):
        if not forecast_df.empty:
            forecast_df = forecast_df.sort_values('yyyyqq')
            # Filter by cutoff date and validate scenario forecast PD values
            forecast_df_filtered = forecast_df[forecast_df['yyyyqq'] < cutoff_date]
            valid_scenario = forecast_df_filtered['predicted_PD'].between(1e-8, 2.0)
            if valid_scenario.sum() > 0:
                plot_data = forecast_df_filtered[valid_scenario]
                
                # Connect forecast to last historical point if available
                if last_hist_date is not None and last_hist_pd is not None and not plot_data.empty:
                    # Create connected line by prepending the last historical point
                    connection_dates = [last_hist_date] + plot_data['yyyyqq'].tolist()
                    connection_values = [last_hist_pd] + (plot_data['predicted_PD'] * 100).tolist()
                    
                    ax1.plot(connection_dates, connection_values, 
                             label=f'{scenario} Forecast', color=colors[i % len(colors)], 
                             linestyle='--', linewidth=3)
                else:
                    # Fallback to disconnected plot if no historical connection available
                    ax1.plot(plot_data['yyyyqq'], plot_data['predicted_PD'] * 100, 
                             label=f'{scenario} Forecast', color=colors[i % len(colors)], 
                             linestyle='--', linewidth=3)
    
    # Mark forecast start
    ax1.axvline(pd.to_datetime(FORECAST_START), color='gray', linestyle=':', 
                label='Forecast Start', alpha=0.7, linewidth=2)
    
    # Get model description using helper function
    ovs_source = ovs_model.get('ovs_source', 'filtered')
    lag_option = ovs_model.get('lag_option', 'no_lags')
    model_desc = get_model_description_for_plotting(country, ovs_source, lag_option)
    
    ax1.set_title(f'PD Scenario Forecasts: {country}\n'
                  f'{model_desc}\n'
                  f'Adjusted R-squared: {ovs_model["adj_r2"]*100:.2f}%', fontsize=15)
    ax1.set_ylabel('Probability of Default (%)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits using linear scale only (now in percentage)
    if not hist_pd_data.empty:
        # Collect all PD values for better range calculation (convert to percentage)
        all_pd_values = []
        
        # Add historical PD values
        all_pd_values.extend((hist_pd_data['cdsiedf5'] * 100).dropna().tolist())
        
        # Add historical forecast PD values if available (with more lenient validation)
        if historical_forecast is not None and not historical_forecast.empty:
            historical_forecast_filtered = historical_forecast[
                historical_forecast['yyyyqq'] < cutoff_date
            ]
            valid_hist_forecast = historical_forecast_filtered['predicted_PD'].between(1e-8, 2.0)
            if valid_hist_forecast.sum() > 0:
                hist_forecast_values = (historical_forecast_filtered.loc[valid_hist_forecast, 'predicted_PD'] * 100).tolist()
                all_pd_values.extend(hist_forecast_values)
        
        # Add scenario forecast PD values (with more lenient validation)
        for scenario, forecast_df in scenario_forecasts.items():
            if not forecast_df.empty:
                forecast_df_filtered = forecast_df[forecast_df['yyyyqq'] < cutoff_date]
                valid_scenario = forecast_df_filtered['predicted_PD'].between(1e-8, 2.0)
                if valid_scenario.sum() > 0:
                    scenario_values = (forecast_df_filtered.loc[valid_scenario, 'predicted_PD'] * 100).tolist()
                    all_pd_values.extend(scenario_values)
        
        if all_pd_values:
            pd_min, pd_max = min(all_pd_values), max(all_pd_values)
            
            # Always use linear scale with adaptive padding
            pd_range = pd_max - pd_min
            
            # Use larger padding for very small ranges
            if pd_range < pd_max * 0.1:  # Range is less than 10% of max value
                padding = pd_max * 0.2  # Use 20% of max value as padding
            else:
                padding = max(pd_range * 0.2, pd_max * 0.1)  # 20% of range or 10% of max
            
            y_min = max(pd_min - padding, 0)  # PD can't be negative
            y_max = min(pd_max + padding, 200.0)  # Allow some room above 100% for forecast errors
            
            # Ensure minimum range for very flat data
            if y_max - y_min < pd_max * 0.1:
                center = (y_min + y_max) / 2
                half_range = pd_max * 0.1
                y_min = max(center - half_range, 0)
                y_max = min(center + half_range, 200.0)
            
            ax1.set_ylim(y_min, y_max)
        else:
            # Fallback if no valid PD values found
            ax1.set_ylim(0, 10)
    else:
        # Fallback if no historical data
        ax1.set_ylim(0, 10)
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))
    
    # Set x-axis limits to show data before 2040
    ax1.set_xlim(left=None, right=cutoff_date)
    
    # Add x-label to the PD plot since it's now the only plot
    ax1.set_xlabel('Date', fontsize=14)
    
    # Increase tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(plots_dir / f'scenario_forecast_{country}_{combination_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib rcParams to default
    plt.rcdefaults()

def plot_macro_variables(country, historical_data, ovs_model, plots_dir=None, combination_name=""):
    """Plot macro variables in separate subplots, showing only data after PD start"""
    if plots_dir is None:
        plots_dir = Path('Output/8.scenario_forecast/plots')
    
    # Set larger font sizes for all plot elements
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    try:
        # Load all scenario data to show macro variable patterns
        data = pd.read_csv(DATA_PATH)
        data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
        country_data = data[data['cinc'] == country].sort_values('yyyyqq')
        
        # Create lagged variables if needed for plotting
        lag_option = ovs_model.get('lag_option', 'no_lags')
        if lag_option == 'with_lags':
            country_data = create_lagged_variables(country_data, ovs_model)
        
        # Get PD start date - find first non-null PD value
        pd_start_date = None
        if 'cdsiedf5' in historical_data.columns:
            pd_data = historical_data[historical_data['cdsiedf5'].notna()]
            if not pd_data.empty:
                pd_start_date = pd_data['yyyyqq'].min()
        
        # If no PD start date found, use forecast start as fallback
        if pd_start_date is None:
            pd_start_date = pd.to_datetime(FORECAST_START)
        
        # Filter data to show only after PD start and before 2040
        cutoff_date = pd.to_datetime('2040-01-01')
        country_data = country_data[
            (country_data['yyyyqq'] >= pd_start_date) & 
            (country_data['yyyyqq'] < cutoff_date)
        ]
        
        # Get macro variables using helper function
        ovs_source = ovs_model.get('ovs_source', 'filtered')
        macro_vars = get_macro_variables_for_plotting(country, ovs_source, lag_option)
        
        if not macro_vars:
            print(f"  No macro variables found for {country}")
            return
        
        # Create subplots - one for each macro variable
        n_vars = len(macro_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=(16, 4 * n_vars))
        
        # Handle case where there's only one subplot
        if n_vars == 1:
            axes = [axes]
        
        scenarios = ['Baseline', 'S1', 'S3', 'S4']
        colors = ['blue', 'orange', 'red', 'green', 'purple']
        
        for j, mv in enumerate(macro_vars):
            ax = axes[j]
            
            # Clean variable name for display (remove lag suffix for title)
            clean_mv_name = mv.replace('_Baseline_trans', '')
            
            # Plot each scenario for this macro variable
            for i, scenario in enumerate(scenarios):
                mv_scenario_name = f"{mv}_{scenario}_trans"
                
                if mv_scenario_name in country_data.columns:
                    # Plot data
                    valid_data = country_data[country_data[mv_scenario_name].notna()]
                    if not valid_data.empty:
                        linestyle = '-' if scenario == 'Baseline' else '--'
                        alpha = 1.0 if scenario == 'Baseline' else 0.7
                        linewidth = 3 if scenario == 'Baseline' else 2
                        
                        ax.plot(valid_data['yyyyqq'], valid_data[mv_scenario_name], 
                               color=colors[i % len(colors)], linestyle=linestyle, 
                               alpha=alpha, linewidth=linewidth, label=scenario)
            
            # Mark forecast start
            ax.axvline(pd.to_datetime(FORECAST_START), color='gray', linestyle=':', 
                      label='Forecast Start', alpha=0.7, linewidth=2)
            
            # Mark PD start if different from forecast start
            if pd_start_date != pd.to_datetime(FORECAST_START):
                ax.axvline(pd_start_date, color='lightgray', linestyle=':', 
                          label='PD Start', alpha=0.7, linewidth=2)
            
            ax.set_title(f'{clean_mv_name}', fontsize=16)
            ax.set_ylabel('Standardized Value', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits to show data before 2040
            ax.set_xlim(left=None, right=cutoff_date)
            
            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Only add x-label to bottom subplot
            if j == n_vars - 1:
                ax.set_xlabel('Date', fontsize=14)
        
        # Main title - include lag information
        lag_info = " (with lags)" if lag_option == 'with_lags' else " (no lags)"
        fig.suptitle(f'Macro Variables for {country}{lag_info} (After PD Start: {pd_start_date.strftime("%Y-%m-%d")})', 
                     fontsize=18, y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plots_dir / f'macro_variables_{country}_{combination_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved macro variables plot for {country} ({lag_option})")
        
    except Exception as e:
        print(f"  Error creating macro variables plot for {country}: {str(e)}")
        # Create empty plot with error message
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.text(0.5, 0.5, f'Error loading macro variable data: {str(e)}', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Macro Variables for {country} (Error)', fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / f'macro_variables_{country}_{combination_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Reset matplotlib rcParams to default
    plt.rcdefaults()

def calculate_forecast_metrics(historical_forecast):
    """Calculate forecast accuracy metrics for historical 1-step forecasts"""
    if historical_forecast is None or historical_forecast.empty:
        return {}
    
    # Remove NaN values
    valid_data = historical_forecast.dropna(subset=['predicted_PD', 'actual_PD', 'predicted_dlnPD', 'actual_dlnPD'])
    
    if len(valid_data) == 0:
        return {}
    
    # PD level metrics
    pd_mae = np.mean(np.abs(valid_data['predicted_PD'] - valid_data['actual_PD']))
    pd_rmse = np.sqrt(np.mean((valid_data['predicted_PD'] - valid_data['actual_PD'])**2))
    pd_mape = np.mean(np.abs((valid_data['predicted_PD'] - valid_data['actual_PD']) / valid_data['actual_PD'])) * 100
    
    # dlnPD metrics
    dlnpd_mae = np.mean(np.abs(valid_data['predicted_dlnPD'] - valid_data['actual_dlnPD']))
    dlnpd_rmse = np.sqrt(np.mean((valid_data['predicted_dlnPD'] - valid_data['actual_dlnPD'])**2))
    
    # Correlation
    pd_corr = valid_data['predicted_PD'].corr(valid_data['actual_PD'])
    dlnpd_corr = valid_data['predicted_dlnPD'].corr(valid_data['actual_dlnPD'])
    
    return {
        'n_observations': len(valid_data),
        'pd_mae': pd_mae,
        'pd_rmse': pd_rmse,
        'pd_mape': pd_mape,
        'pd_correlation': pd_corr,
        'dlnpd_mae': dlnpd_mae,
        'dlnpd_rmse': dlnpd_rmse,
        'dlnpd_correlation': dlnpd_corr
    }

def generate_forecast(country, model_vars, coefficients, scenario_data, scenario_name, periods=20):
    """Generate forecast for a country using OVS model"""
    
    try:
        # Get country data
        country_data = scenario_data[scenario_data['country'] == country].copy()
        if country_data.empty:
            print(f"Warning: No data found for {country}")
            return None
        
        # Sort by date and get the most recent data point
        country_data = country_data.sort_values('date')
        latest_data = country_data.iloc[-1].copy()
        
        # Initialize forecast
        forecast_results = []
        current_data = latest_data.copy()
        
        # Convert coefficients list to dictionary for easier access
        coeff_dict = dict(zip(model_vars, coefficients))
        
        for period in range(1, periods + 1):
            # Calculate forecast
            forecast_value = 0.0
            
            # Add constant term
            if 'const' in coeff_dict:
                forecast_value += coeff_dict['const']
            
            # Add dlnPD lag term (only lag 1 in the OVS models)
            if 'dlnPD_l1' in coeff_dict:
                if period == 1:
                    # For first period, use actual last dlnPD
                    lag_value = current_data.get('dlnPD', 0)
                else:
                    # Use forecasted value from previous period
                    lag_value = forecast_results[-1]['dlnPD_forecast']
                
                forecast_value += coeff_dict['dlnPD_l1'] * lag_value
            
            # Add mean reverting term
            if 'mean_reverting' in coeff_dict:
                # Mean reverting term typically uses lagged PD level
                mean_reverting_value = current_data.get('mean_reverting', 0)
                forecast_value += coeff_dict['mean_reverting'] * mean_reverting_value
            
            # Add macro variable contributions
            for var in model_vars:
                if var not in ['const', 'dlnPD_l1', 'mean_reverting']:
                    # This is a macro variable
                    if var in current_data:
                        forecast_value += coeff_dict[var] * current_data[var]
                    else:
                        print(f"Warning: Variable {var} not found in data for {country}")
            
            # Store forecast result
            forecast_result = {
                'country': country,
                'scenario': scenario_name,
                'period': period,
                'date': pd.to_datetime(current_data['date']) + pd.DateOffset(months=3*period),
                'dlnPD_forecast': forecast_value,
                'PD_forecast': None  # Will be calculated after all dlnPD forecasts
            }
            
            forecast_results.append(forecast_result)
            
            # Update current_data for next iteration (for lag terms)
            current_data['dlnPD'] = forecast_value
        
        # Calculate PD forecasts from dlnPD forecasts
        initial_pd = current_data.get('PD', 1.0)  # Starting PD level
        current_pd = initial_pd
        
        for i, result in enumerate(forecast_results):
            # PD(t) = PD(t-1) * exp(dlnPD(t))
            current_pd = current_pd * np.exp(result['dlnPD_forecast'])
            forecast_results[i]['PD_forecast'] = current_pd
        
        return pd.DataFrame(forecast_results)
        
    except Exception as e:
        print(f"Error generating forecast for {country}: {str(e)}")
        return None

def plot_scenario_comparison_combo(country, historical_data, scenario_forecasts, ovs_model, historical_forecast=None, 
                                   plots_dir=None, combination_name=""):
    """Plot comparison of scenarios for a country with combination-specific output directory"""
    if plots_dir is None:
        plots_dir = plots_dir
    
    return plot_scenario_comparison(country, historical_data, scenario_forecasts, ovs_model, historical_forecast, plots_dir, combination_name)

def plot_macro_variables_combo(country, historical_data, ovs_model, plots_dir=None, combination_name=""):
    """Plot macro variables for a country with combination-specific output directory"""
    if plots_dir is None:
        plots_dir = plots_dir
    
    return plot_macro_variables(country, historical_data, ovs_model, plots_dir, combination_name)

def main():
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    
    print("Identifying scenarios...")
    scenarios = identify_scenarios(data)
    
    # Prepare data based on forecast cutoff
    forecast_start_dt = pd.to_datetime(FORECAST_START)
    historical_data = data[data['yyyyqq'] < forecast_start_dt]
    forecast_data = data[data['yyyyqq'] >= forecast_start_dt]
    
    # Initialize master summary for all combinations
    all_combinations_summary = []
    all_combinations_metrics = {}
    
    # Process all combinations of OVS sources and lag options
    for ovs_source in OVS_SOURCES.keys():
        for lag_option in LAG_OPTIONS.keys():
            combination_name = f"{ovs_source}_{lag_option}"
            print(f"\nProcessing: {combination_name}")
            
            try:
                ovs_models = load_ovs_results(ovs_source, lag_option)
        
                all_forecast_results = {}
                all_historical_forecasts = {}
                forecast_metrics = {}
                
                # Process each country
                for country, model_info in ovs_models.items():
                    try:
                        # Create country-specific output directory
                        country_output_dir = output_dir / country
                        country_output_dir.mkdir(parents=True, exist_ok=True)
                        country_plots_dir = country_output_dir / 'plots'
                        country_plots_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Get country-specific data
                        country_hist = historical_data[historical_data['cinc'] == country]
                        country_forecast = forecast_data[forecast_data['cinc'] == country]
                        
                        if country_hist.empty:
                            continue
                        
                        # Create lagged variables if needed
                        if model_info.get('lag_option') == 'with_lags':
                            country_hist = create_lagged_variables(country_hist, model_info)
                            country_forecast = create_lagged_variables(country_forecast, model_info)
                        
                        # Generate historical 1-step forecasts for validation
                        historical_forecast = forecast_historical_1step(country_hist, model_info)
                        all_historical_forecasts[country] = historical_forecast
                        
                        # Calculate forecast metrics
                        metrics = calculate_forecast_metrics(historical_forecast)
                        if metrics:
                            forecast_metrics[country] = metrics
                        
                        # Save historical forecasts with combination name
                        if not historical_forecast.empty:
                            historical_forecast.to_csv(country_output_dir / f'historical_forecast_{combination_name}.csv', index=False)
                        
                        # Prepare scenario data
                        scenario_data = prepare_scenario_data(data, country, scenarios, model_info)
                        
                        # Generate forecasts for each scenario
                        scenario_forecasts = {}
                        consolidated_forecast = None
                        
                        for scenario in scenarios:
                            # Get scenario-specific data
                            scenario_hist = scenario_data[scenario][scenario_data[scenario]['yyyyqq'] < forecast_start_dt]
                            scenario_forecast = scenario_data[scenario][scenario_data[scenario]['yyyyqq'] >= forecast_start_dt]
                            
                            if not scenario_forecast.empty:
                                forecast_result = forecast_dlnpd(scenario_hist, scenario_forecast, model_info, scenario)
                                scenario_forecasts[scenario] = forecast_result
                                
                                # Consolidate forecasts into single dataframe
                                if consolidated_forecast is None:
                                    consolidated_forecast = forecast_result[['yyyyqq']].copy()
                                
                                # Add scenario-specific columns
                                consolidated_forecast[f'predicted_dlnPD_{scenario}'] = forecast_result['predicted_dlnPD']
                                consolidated_forecast[f'predicted_PD_{scenario}'] = forecast_result['predicted_PD']
                        
                        # Save consolidated forecast for the country with combination name
                        if consolidated_forecast is not None:
                            consolidated_forecast.to_csv(country_output_dir / f'forecast_all_scenarios_{combination_name}.csv', index=False)
                        
                        # Create comparison plots (save in country-specific directory)
                        if scenario_forecasts:
                            plot_scenario_comparison_combo(country, country_hist, scenario_forecasts, model_info, 
                                                         historical_forecast, country_plots_dir, combination_name)
                            
                            # Copy scenario forecast chart to centralized directory for comparison (advanced filtering + no lags or has_no_lags)
                            if ovs_source in ['advanced', f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q', f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed']:
                                source_chart = country_plots_dir / f'scenario_forecast_{country}_{combination_name}.png'
                                
                                # Copy to appropriate centralized directory based on lag option
                                if lag_option == 'ignore_lags':
                                    target_chart = comparison_plots_dir_ignore_lags / f'scenario_forecast_{country}_{combination_name}.png'
                                elif lag_option == 'has_no_lags':
                                    target_chart = comparison_plots_dir_has_no_lags / f'scenario_forecast_{country}_{combination_name}.png'
                                else:
                                    target_chart = None
                                
                                if target_chart and source_chart.exists():
                                    shutil.copy2(source_chart, target_chart)
                            
                            plot_macro_variables_combo(country, country_hist, model_info, country_plots_dir, combination_name)
                            all_forecast_results[country] = scenario_forecasts
                            
                    except Exception as e:
                        print(f"Error processing {country}: {str(e)}")
                        continue
                
                # Store metrics for this combination
                all_combinations_metrics[combination_name] = forecast_metrics
                
                # Add to master summary
                for country, scenarios_dict in all_forecast_results.items():
                    # Add historical forecast info if available
                    hist_metrics = forecast_metrics.get(country, {})
                    
                    for scenario, forecast_df in scenarios_dict.items():
                        if not forecast_df.empty:
                            all_combinations_summary.append({
                                'combination': combination_name,
                                'ovs_source': ovs_source,
                                'lag_option': lag_option,
                                'country': country,
                                'scenario': scenario,
                                'forecast_periods': len(forecast_df),
                                'final_predicted_pd': forecast_df['predicted_PD'].iloc[-1] if not forecast_df.empty else np.nan,
                                'model_adj_r2': ovs_models[country]['adj_r2'],
                                'historical_pd_mape': hist_metrics.get('pd_mape', np.nan),
                                'historical_pd_correlation': hist_metrics.get('pd_correlation', np.nan),
                                'historical_dlnpd_rmse': hist_metrics.get('dlnpd_rmse', np.nan),
                                'historical_dlnpd_correlation': hist_metrics.get('dlnpd_correlation', np.nan)
                            })
                
                # Print summary statistics for this combination
                if forecast_metrics:
                    print(f"\n{combination_name} Performance Summary:")
                    print(f"  PD MAPE: {pd.DataFrame.from_dict(forecast_metrics, orient='index')['pd_mape'].mean():.1f}%")
                    print(f"  PD Correlation: {pd.DataFrame.from_dict(forecast_metrics, orient='index')['pd_correlation'].mean():.3f}")
                    print(f"  dlnPD RMSE: {pd.DataFrame.from_dict(forecast_metrics, orient='index')['dlnpd_rmse'].mean():.4f}")
                
                print(f"{combination_name} complete: {len(all_forecast_results)} countries processed")
                
            except Exception as e:
                print(f"Error processing combination {combination_name}: {str(e)}")
                continue
    
    # Save master summary and metrics files
    if all_combinations_summary:
        master_summary_df = pd.DataFrame(all_combinations_summary)
        master_summary_df.to_csv(output_dir / 'master_forecast_summary_all_combinations.csv', index=False)
    
    # Save master metrics summary
    if all_combinations_metrics:
        master_metrics_data = []
        for combination_name, metrics_dict in all_combinations_metrics.items():
            ovs_source, lag_option = combination_name.split('_', 1)
            for country, metrics in metrics_dict.items():
                metrics_record = {
                    'combination': combination_name,
                    'ovs_source': ovs_source,
                    'lag_option': lag_option,
                    'country': country,
                    **metrics
                }
                master_metrics_data.append(metrics_record)
        
        master_metrics_df = pd.DataFrame(master_metrics_data)
        master_metrics_df.to_csv(output_dir / 'master_historical_forecast_metrics_all_combinations.csv', index=False)
    
    # Create country-specific summary files
    if all_combinations_summary:
        master_df = pd.DataFrame(all_combinations_summary)
        countries = master_df['country'].unique()
        
        for country in countries:
            country_data = master_df[master_df['country'] == country]
            country_output_dir = output_dir / country
            country_data.to_csv(country_output_dir / f'{country}_all_combinations_summary.csv', index=False)
            
            # Also create country-specific metrics file if available
            if all_combinations_metrics:
                country_metrics_data = []
                for combination_name, metrics_dict in all_combinations_metrics.items():
                    if country in metrics_dict:
                        ovs_source, lag_option = combination_name.split('_', 1)
                        metrics_record = {
                            'combination': combination_name,
                            'ovs_source': ovs_source,
                            'lag_option': lag_option,
                            **metrics_dict[country]
                        }
                        country_metrics_data.append(metrics_record)
                
                if country_metrics_data:
                    country_metrics_df = pd.DataFrame(country_metrics_data)
                    country_metrics_df.to_csv(country_output_dir / f'{country}_all_combinations_metrics.csv', index=False)
    
    print(f"\nALL COMBINATIONS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated 8 combinations: 4 OVS sources × 2 lag options")
    print(f"Master files: summary and metrics CSVs created")
    print(f"Comparison charts (ignore_lags): {comparison_plots_dir_ignore_lags}")
    print(f"Comparison charts (has_no_lags): {comparison_plots_dir_has_no_lags}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 