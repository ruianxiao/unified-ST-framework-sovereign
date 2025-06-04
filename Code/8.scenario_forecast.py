import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings
import re
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
output_dir = Path('Output/8.scenario_forecast')
output_dir.mkdir(parents=True, exist_ok=True)
plots_dir = output_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

def load_regression_models():
    """Load regression models from step 5 output"""
    with open('Output/5.regression_analysis/regression_results.json', 'r') as f:
        models = json.load(f)
    return models

def parse_var_name(var):
    """Parse variable name to extract base variable and lag/lead"""
    if '_lag' in var:
        match = re.match(r'(.*)_lag(\d+)', var)
        if match:
            return match.group(1), int(match.group(2)), 'lag'
    elif '_lead' in var:
        match = re.match(r'(.*)_lead(\d+)', var)
        if match:
            return match.group(1), int(match.group(2)), 'lead'
    return var, 0, None

def prepare_forecast_data(data, country, model_info, forecast_start='2025Q3'):
    """
    Prepare data for forecasting by:
    1. Filtering for the specific country
    2. Creating lagged/leading variables as needed by the model
    3. Separating historical and forecast periods
    """
    # Filter for country
    country_data = data[data['cinc'] == country].copy()
    
    # Get required variables from model
    required_vars = model_info['selected_vars']
    
    # Create lagged/leading variables
    for var in required_vars:
        base_var, shift, shift_type = parse_var_name(var)
        if shift_type == 'lag':
            country_data[var] = country_data[base_var].shift(shift)
        elif shift_type == 'lead':
            country_data[var] = country_data[base_var].shift(-shift)
    
    # Split into historical and forecast periods
    forecast_start = pd.to_datetime(forecast_start)
    historical = country_data[country_data['yyyyqq'] < forecast_start]
    forecast = country_data[country_data['yyyyqq'] >= forecast_start]
    
    return historical, forecast

def forecast_pd(historical, forecast, model_info):
    """
    Recursively forecast PD changes and levels
    """
    # Get model coefficients
    coefs = model_info['coefficients'].copy()  # Make a copy to avoid modifying original
    const = coefs.pop('const')
    
    # Initialize forecast results
    forecast_results = forecast.copy()
    forecast_results['predicted_dlnPD'] = np.nan
    forecast_results['predicted_lnPD'] = np.nan
    
    # Get last historical values for initial conditions
    last_historical = historical.iloc[-1]
    current_lnpd = last_historical['lnPD']
    
    # For each forecast period
    for idx in forecast.index:
        # Prepare X for prediction
        X = pd.Series(index=model_info['selected_vars'], dtype=float)
        for var in model_info['selected_vars']:
            if var == 'lnPD_TTC_gap':
                # Calculate gap using current lnPD and historical mean
                historical_mean = historical['lnPD'].mean()
                X[var] = current_lnpd - historical_mean
            elif var == 'dlnPD_lag':
                # Use last period's predicted change
                if pd.isna(forecast_results.loc[idx, 'predicted_dlnPD']):
                    X[var] = last_historical['dlnPD']
                else:
                    X[var] = forecast_results.loc[idx, 'predicted_dlnPD']
            else:
                # Use forecasted value
                X[var] = forecast.loc[idx, var]
        
        # Predict dlnPD
        predicted_dlnpd = const + sum(coefs[var] * X[var] for var in model_info['selected_vars'])
        forecast_results.loc[idx, 'predicted_dlnPD'] = predicted_dlnpd
        
        # Update lnPD
        current_lnpd += predicted_dlnpd
        forecast_results.loc[idx, 'predicted_lnPD'] = current_lnpd
    
    # Convert log PD to actual PD
    forecast_results['predicted_PD'] = np.exp(forecast_results['predicted_lnPD'])
    
    return forecast_results

def plot_pd_series(country, historical, forecast_results, forecast_start):
    """Plot historical and forecasted PD for a country"""
    plt.figure(figsize=(10, 5))
    # Historical
    plt.plot(historical['yyyyqq'], np.exp(historical['lnPD']), label='Historical PD', color='blue')
    # Forecasted
    plt.plot(forecast_results['yyyyqq'], forecast_results['predicted_PD'], label='Forecasted PD', color='red', linestyle='--')
    # Mark forecast start
    plt.axvline(pd.to_datetime(forecast_start), color='black', linestyle=':', label='Forecast Start')
    plt.title(f'PD Forecast for {country}')
    plt.xlabel('Date')
    plt.ylabel('PD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f'PD_forecast_{country}.png')
    plt.close()

def main():
    # Load data
    logger.info("Loading data...")
    data = pd.read_csv('Output/4.transformation/transformed_data.csv')
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    
    # Load regression models
    logger.info("Loading regression models...")
    models = load_regression_models()
    
    # Initialize results storage
    all_forecasts = {}
    forecast_start = '2025-07-01'  # 2025Q3 as datetime
    
    # Process each country
    for country, model_info in models.items():
        logger.info(f"Processing {country}...")
        try:
            # Prepare data
            historical, forecast = prepare_forecast_data(data, country, model_info, forecast_start=forecast_start)
            
            # Generate forecasts
            forecast_results = forecast_pd(historical, forecast, model_info)
            
            # Store results
            all_forecasts[country] = {
                'forecast_periods': forecast_results['yyyyqq'].dt.strftime('%Y-%m-%d').tolist(),
                'predicted_PD': forecast_results['predicted_PD'].tolist(),
                'predicted_dlnPD': forecast_results['predicted_dlnPD'].tolist(),
                'model_variables': model_info['selected_vars'],
                'model_coefficients': model_info['coefficients']
            }
            
            # Save individual country results
            forecast_results.to_csv(output_dir / f'forecast_{country}.csv', index=False)
            
            # Plot historical + forecasted PD
            plot_pd_series(country, historical, forecast_results, forecast_start)
            
        except Exception as e:
            logger.error(f"Error processing {country}: {str(e)}")
            continue
    
    # Save summary of all forecasts
    with open(output_dir / 'forecast_summary.json', 'w') as f:
        json.dump(all_forecasts, f, indent=2)
    
    logger.info("Forecasting complete! Results saved in Output/8.scenario_forecast/")

if __name__ == "__main__":
    main() 