import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
from pathlib import Path
import ast
import warnings
import re
warnings.filterwarnings('ignore')

# Output directories
output_dir = Path('Output/8.1.scenario_forecast_top_models')
output_dir.mkdir(parents=True, exist_ok=True)

# Configuration
DATA_PATH = 'Output/4.transformation/transformed_data.csv'
GCORR_FORECAST_QUARTERS = 40  # Number of GCorr forecast quarters (20 or 40)
FILTERED_RESULTS_FILE = f'Output/7.filtered_ovs_results/filtered_ovs_results_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed.csv'
FORECAST_START = '2025-07-01'  # 2025Q3

def load_all_models_for_country(country):
    """
    Load all filtered models for a specific country from filtered results.
    Returns list of model dictionaries or empty list if not found.
    """
    try:
        filtered_data = pd.read_csv(FILTERED_RESULTS_FILE)
        country_models = filtered_data[filtered_data['country'] == country]
        
        if country_models.empty:
            print(f"  No models found for {country}")
            return []
        
        # Get all models (they're already ranked in filtered results)
        all_models = country_models
        
        models_list = []
        for idx, model in all_models.iterrows():
            # Build model_vars and coefficients
            model_vars = []
            coefficients = []
            
            # Add intercept
            if 'constant_coefficient' in model.index and pd.notna(model['constant_coefficient']):
                model_vars.append('const')
                coefficients.append(model['constant_coefficient'])
            
            # Add dlnPD lags
            if 'includes_lag' in model.index and model['includes_lag'] and pd.notna(model['lag_coefficient']):
                model_vars.append('dlnPD_l1')
                coefficients.append(model['lag_coefficient'])
            
            # Add mean reverting term
            if 'mean_reverting_coefficient' in model.index and pd.notna(model['mean_reverting_coefficient']):
                model_vars.append('mean_reverting')
                coefficients.append(model['mean_reverting_coefficient'])
            
            # Add macro variables
            for i in range(1, 5):  # MV1-MV4
                mv_col = f'MV{i}'
                coeff_col = f'MV{i}_coefficient'
                
                if mv_col in model.index and pd.notna(model[mv_col]) and model[mv_col].strip():
                    base_mv_name = model[mv_col].split('_lag')[0] if '_lag' in model[mv_col] else model[mv_col]
                    mv_name = f"{base_mv_name}_Baseline_trans"
                    
                    model_vars.append(mv_name)
                    coefficients.append(model[coeff_col])
            
            model_info = {
                'model_vars': model_vars,
                'coefficients': coefficients,
                'adj_r2': model['adj_r2'],
                'rank': model['rank_in_country'],
                'has_no_lags': model.get('has_no_lags', False),
                'country': country
            }
            
            models_list.append(model_info)
        
        print(f"  Loaded {len(models_list)} models for {country}")
        return models_list
        
    except Exception as e:
        print(f"  Error loading models for {country}: {str(e)}")
        return []

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

def prepare_scenario_data(data, country, scenarios):
    """Prepare data for each scenario"""
    scenario_data = {}
    
    # Filter for specific country
    country_data = data[data['cinc'] == country].copy().sort_values('yyyyqq')
    
    for scenario in scenarios:
        scenario_df = country_data.copy()
        scenario_data[scenario] = scenario_df
    
    return scenario_data

def map_model_vars_to_scenario(model_vars, scenario):
    """Map model variable names from Baseline to scenario-specific names"""
    if scenario == 'Baseline':
        return model_vars
    
    mapped_vars = []
    for var in model_vars:
        if '_Baseline_trans' in var:
            mapped_var = var.replace('_Baseline_trans', f'_{scenario}_trans')
            mapped_vars.append(mapped_var)
        else:
            # Keep non-baseline variables as is (dlnPD_l1, mean_reverting, const)
            mapped_vars.append(var)
    
    return mapped_vars

def forecast_dlnpd(historical_data, forecast_data, model_info, scenario='Baseline'):
    """Forecast dlnPD using the OVS model"""
    model_vars = model_info['model_vars']
    coefficients = model_info['coefficients']
    
    # Convert coefficients list to dictionary for easier access
    coeff_dict = dict(zip(model_vars, coefficients))
    
    # Map model variables to scenario-specific names
    scenario_model_vars = map_model_vars_to_scenario(model_vars, scenario)
    
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

def process_country_all_models(country, data, scenarios):
    """Process all filtered models for a specific country"""
    print(f"\nProcessing {country}...")
    
    # Create country-specific output directory
    country_output_dir = output_dir / country
    country_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all filtered models for this country
    models = load_all_models_for_country(country)
    
    if not models:
        print(f"  No models available for {country}")
        return
    
    # Get country-specific data
    forecast_start_dt = pd.to_datetime(FORECAST_START)
    historical_data = data[data['cinc'] == country]
    historical_data = historical_data[historical_data['yyyyqq'] < forecast_start_dt]
    forecast_data = data[data['cinc'] == country]
    forecast_data = forecast_data[forecast_data['yyyyqq'] >= forecast_start_dt]
    
    if historical_data.empty:
        print(f"  No historical data for {country}")
        return
    
    # Prepare scenario data
    scenario_data = prepare_scenario_data(data, country, scenarios)
    
    # Process each model
    for model_idx, model_info in enumerate(models):
        model_rank = model_info['rank']
        print(f"  Processing model rank {model_rank} (adj_r2: {model_info['adj_r2']:.3f})")
        
        try:
            # Generate forecasts for each scenario
            consolidated_forecast = None
            
            for scenario in scenarios:
                # Get scenario-specific data
                scenario_hist = scenario_data[scenario][scenario_data[scenario]['yyyyqq'] < forecast_start_dt]
                scenario_forecast = scenario_data[scenario][scenario_data[scenario]['yyyyqq'] >= forecast_start_dt]
                
                if not scenario_forecast.empty:
                    forecast_result = forecast_dlnpd(scenario_hist, scenario_forecast, model_info, scenario)
                    
                    # Consolidate forecasts into single dataframe
                    if consolidated_forecast is None:
                        consolidated_forecast = forecast_result[['yyyyqq']].copy()
                    
                    # Add scenario-specific columns
                    consolidated_forecast[f'predicted_dlnPD_{scenario}'] = forecast_result['predicted_dlnPD']
                    consolidated_forecast[f'predicted_PD_{scenario}'] = forecast_result['predicted_PD']
            
            # Save consolidated forecast for this specific model
            if consolidated_forecast is not None:
                # Save individual model forecast file
                model_filename = f'forecast_model_rank_{model_rank}_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags.csv'
                consolidated_forecast.to_csv(country_output_dir / model_filename, index=False)
                
                # Also save the combined forecast file (using top model for compatibility)
                if model_rank == 1:
                    combined_filename = f'forecast_all_scenarios_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags.csv'
                    consolidated_forecast.to_csv(country_output_dir / combined_filename, index=False)
                
                print(f"    Saved forecast for model rank {model_rank}")
        
        except Exception as e:
            print(f"    Error processing model rank {model_rank}: {str(e)}")
            continue
    
    print(f"  Completed {country}: processed {len(models)} models")

def main():
    print("Step 8.1: All Filtered Models Scenario Forecast Generation")
    print(f"Output Directory: {output_dir}")
    print(f"Using filtered results: {FILTERED_RESULTS_FILE}")
    print("=" * 80)
    
    # Check if required files exist
    print("Checking required files...")
    if not Path(DATA_PATH).exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    if not Path(FILTERED_RESULTS_FILE).exists():
        print(f"ERROR: Filtered results file not found: {FILTERED_RESULTS_FILE}")
        return
    print("All required files found.")
    
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    
    print("Identifying scenarios...")
    scenarios = identify_scenarios(data)
    
    # Get list of countries from filtered results
    print("Loading country list from filtered results...")
    filtered_data = pd.read_csv(FILTERED_RESULTS_FILE)
    countries = sorted(filtered_data['country'].unique())
    print(f"Found {len(countries)} countries in filtered results")
    
    # Process each country
    successful_countries = 0
    total_models_processed = 0
    
    for i, country in enumerate(countries, 1):
        try:
            print(f"\n[{i}/{len(countries)}] Processing {country}...")
            
            # Load models for this country
            models = load_all_models_for_country(country)
            if models:
                process_country_all_models(country, data, scenarios)
                successful_countries += 1
                total_models_processed += len(models)
            else:
                print(f"  Skipping {country}: no valid models")
                
        except Exception as e:
            print(f"  Error processing {country}: {str(e)}")
            continue
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("STEP 8.1 COMPLETE!")
    print(f"Successfully processed: {successful_countries}/{len(countries)} countries")
    print(f"Total models processed: {total_models_processed}")
    print(f"Average models per country: {total_models_processed/successful_countries:.1f}" if successful_countries > 0 else "N/A")
    print(f"Results saved to: {output_dir}")
    print(f"Focus: GCorr smoothed version only")
    print(f"Scenarios: {scenarios}")
    
    # Create master summary file
    summary_data = []
    for country_dir in output_dir.iterdir():
        if country_dir.is_dir():
            country = country_dir.name
            model_files = list(country_dir.glob(f'forecast_model_rank_*_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags.csv'))
            
            summary_data.append({
                'country': country,
                'models_generated': len(model_files),
                'model_ranks': [int(f.stem.split('_')[3]) for f in model_files]
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'processing_summary.csv', index=False)
        print(f"Processing summary saved to: {output_dir / 'processing_summary.csv'}")
    
    print("Individual model forecast files are now available for step 9.1!")
    print("Note: All filtered models (not just top 10) have been processed.")

if __name__ == "__main__":
    main()
