"""
Step 9.1: Compare Top 10 OVS Models vs GCorr Forecasts
Compare top 10 individual OVS models with GCorr forecasts for S3 and S4 scenarios.
This script is configurable between 20Q and 40Q GCorr forecast horizons.

Configuration:
- Set GCORR_FORECAST_QUARTERS to 20 or 40 to match the desired forecast horizon
- Uses corresponding OVS models trained on the same GCorr data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
GCORR_FORECAST_QUARTERS = 40  # Number of GCorr forecast quarters (20 or 40)
HISTORICAL_DATA_FILE = 'Output/4.transformation/transformed_data.csv'
OVS_FORECAST_DIR = Path('Output/8.1.scenario_forecast_top_models')  # Updated to use 8.1 output
GCORR_FORECAST_DIR = Path('gcorr-research-delivery-validation/Output/gcorr_scenario_plots/annualized_data')
FORECAST_START = '2025-07-01'

# Output directories
OUTPUT_DIR = Path(f'Output/9.1.ovs_gcorr_top_models_comparison_{GCORR_FORECAST_QUARTERS}q')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model combinations to use from OVS (only smoothed)
OVS_COMBINATIONS = {
    'ovs_gcorr_smoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags'  # Only OVS trained on GCorr smoothed data
}

# GCorr data types to compare (only smoothed)
GCORR_TYPES = {
    'smoothed': 'smoothed'
}

# For testing - limit number of countries (set to None for all countries)
TEST_LIMIT = None

def load_historical_data():
    """Load historical data for all countries"""
    print("Loading historical data...")
    hist_data = pd.read_csv(HISTORICAL_DATA_FILE)
    hist_data['yyyyqq'] = pd.to_datetime(hist_data['yyyyqq'])
    
    # Filter to before forecast start
    forecast_start_dt = pd.to_datetime(FORECAST_START)
    hist_data = hist_data[hist_data['yyyyqq'] < forecast_start_dt]
    
    return hist_data

def get_historical_pd_for_country(historical_data, country):
    """Get historical PD data for a specific country"""
    country_hist = historical_data[historical_data['cinc'] == country].copy()
    
    if country_hist.empty:
        return None
    
    # Filter for valid PD data
    pd_data = country_hist[country_hist['cdsiedf5'].notna() & (country_hist['cdsiedf5'] > 0)]
    
    if pd_data.empty:
        return None
    
    # Sort by date
    pd_data = pd_data.sort_values('yyyyqq')
    
    return pd_data[['yyyyqq', 'cinc', 'cdsiedf5']].rename(columns={'cdsiedf5': 'historical_pd'})

def load_ovs_all_forecasts(country):
    """Load all OVS forecast models for a specific country from step 8.1 output"""
    ovs_forecasts = {}
    
    # Check if country directory exists in step 8.1 output
    country_dir = OVS_FORECAST_DIR / country
    if not country_dir.exists():
        print(f"  No OVS forecast directory found for {country}")
        return ovs_forecasts
    
    # Look for individual model forecast files
    model_files = list(country_dir.glob(f'forecast_model_rank_*_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags.csv'))
    
    if not model_files:
        print(f"  No individual model forecast files found for {country}")
        return ovs_forecasts
    
    # Sort files by rank number
    model_files.sort(key=lambda x: int(x.stem.split('_')[3]))
    
    # Load each model's forecast
    for forecast_file in model_files:
        try:
            # Extract rank from filename
            model_rank = int(forecast_file.stem.split('_')[3])
            
            ovs_data = pd.read_csv(forecast_file)
            ovs_data['yyyyqq'] = pd.to_datetime(ovs_data['yyyyqq'])
            
            # Rename columns to include rank information
            ovs_renamed = ovs_data.rename(columns={
                'predicted_PD_S3': f'ovs_rank{model_rank}_PD_S3',
                'predicted_PD_S4': f'ovs_rank{model_rank}_PD_S4'
            })
            
            ovs_forecasts[f'ovs_rank{model_rank}'] = ovs_renamed
            
        except Exception as e:
            print(f"    Error loading {forecast_file.name} for {country}: {e}")
            continue
    
    print(f"  Loaded {len(ovs_forecasts)} OVS model forecasts for {country}")
    return ovs_forecasts

def load_gcorr_forecasts(country):
    """Load smoothed GCorr forecast data for a specific country"""
    gcorr_forecasts = {}
    
    # Only load smoothed data
    gcorr_file = GCORR_FORECAST_DIR / f'{country}_annualized_smoothed_pd.csv'
    
    if not gcorr_file.exists():
        print(f"  GCorr forecast file not found for {country}: {gcorr_file.name}")
        return gcorr_forecasts
    
    try:
        gcorr_data = pd.read_csv(gcorr_file)
        gcorr_data['date'] = pd.to_datetime(gcorr_data['date'])
        
        # Pivot to have scenarios as columns
        gcorr_pivot = gcorr_data.pivot_table(
            index='date', 
            columns='scenario', 
            values='annual_pd', 
            aggfunc='first'
        ).reset_index()
        
        # Rename columns to distinguish from OVS
        scenario_columns = [col for col in gcorr_pivot.columns if col != 'date']
        rename_dict = {col: f'gcorr_smoothed_PD_{col}' for col in scenario_columns}
        gcorr_pivot = gcorr_pivot.rename(columns=rename_dict)
        gcorr_pivot = gcorr_pivot.rename(columns={'date': 'yyyyqq'})
        
        gcorr_forecasts['smoothed'] = gcorr_pivot
        
    except Exception as e:
        print(f"  Error loading GCorr forecast for {country}: {e}")
    
    return gcorr_forecasts

def get_final_selected_model_rank(country):
    """Get the rank of the final selected model from final_top_models file"""
    try:
        final_models_file = Path(f'Output/7.filtered_ovs_results/final_top_models_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed.csv')
        if final_models_file.exists():
            final_models = pd.read_csv(final_models_file)
            country_model = final_models[final_models['country'] == country]
            if not country_model.empty:
                return country_model.iloc[0]['rank_in_country']
    except Exception as e:
        print(f"  Warning: Could not load final selected model rank for {country}: {e}")
    
    return None  # Return None if not found

def get_country_metadata(country):
    """Get metadata about the country from both models"""
    metadata = {'country': country}
    
    # Get OVS metadata from step 7 filtered results files
    try:
        ovs_detailed_file = Path(f'Output/7.filtered_ovs_results/filtered_ovs_results_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed.csv')
        if ovs_detailed_file.exists():
            ovs_detailed = pd.read_csv(ovs_detailed_file)
            country_results = ovs_detailed[ovs_detailed['country'] == country]
            if not country_results.empty:
                # Get all models info
                all_models = country_results
                metadata['ovs_num_models'] = len(all_models)
                metadata['ovs_best_adj_r2'] = all_models.iloc[0]['adj_r2']
                metadata['ovs_worst_adj_r2'] = all_models.iloc[-1]['adj_r2']
                
                # Get final selected model rank for emphasis
                final_rank = get_final_selected_model_rank(country)
                metadata['final_selected_rank'] = final_rank
    except Exception as e:
        print(f"  Warning: Could not load OVS metadata for {country}: {e}")
    
    # Try to get GCorr metadata
    try:
        gcorr_summary_file = Path('gcorr-research-delivery-validation/Output/gcorr_scenario_plots/gcorr_scenario_forecast_summary_annualized.csv')
        if gcorr_summary_file.exists():
            gcorr_summary = pd.read_csv(gcorr_summary_file)
            gcorr_country = gcorr_summary[gcorr_summary['country'] == country]
            if not gcorr_country.empty:
                for _, row in gcorr_country.iterrows():
                    metadata['gcorr_rsq'] = row['rsq']
                    metadata['gcorr_mv_variables'] = row['mv_variables']
                    metadata['gcorr_num_mv'] = row['num_mv']
                    break  # Take first row
    except Exception as e:
        print(f"  Warning: Could not load GCorr metadata for {country}: {e}")
    
    return metadata

def create_comparison_plots(country, historical_data, ovs_forecasts, gcorr_forecasts, metadata):
    """Create comparison plots for a country - only smoothed, S3 and S4 scenarios"""
    
    # Only create smoothed comparison
    if gcorr_forecasts.get('smoothed') is not None:
        create_single_comparison_plot(country, 'smoothed', historical_data, ovs_forecasts, gcorr_forecasts, metadata)

def create_single_comparison_plot(country, gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, metadata):
    """Create comparison plot for a country with all OVS models vs GCorr (S3 and S4 only)"""
    
    # Set larger font sizes for all plot elements
    plt.rcParams.update({
        'font.size': 18,           # Base font size
        'axes.titlesize': 22,      # Subplot titles
        'axes.labelsize': 22,      # Axis labels (increased)
        'xtick.labelsize': 18,     # X-axis tick labels (increased)
        'ytick.labelsize': 18,     # Y-axis tick labels (increased)
        'legend.fontsize': 16,     # Legend text (smaller for more models)
        'figure.titlesize': 24     # Main title
    })
    
    # Set up the plot with 1x2 layout (only S3 and S4)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f'All OVS Models vs GCorr Forecast: {country} (GCorr {GCORR_FORECAST_QUARTERS}Q Smoothed)', fontsize=24, y=0.96)
    
    # Define color palette for all OVS models (use tab20 for more colors)
    num_models = len(ovs_forecasts)
    if num_models <= 10:
        ovs_colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    else:
        ovs_colors = plt.cm.tab20(np.linspace(0, 1, min(num_models, 20)))
    gcorr_color = 'red'  # Keep GCorr as red for consistency
    
    # Get final selected model rank for emphasis
    final_selected_rank = metadata.get('final_selected_rank')
    
    # Define scenario mappings (GCorr uses 'S3', 'S4' directly)
    scenario_mapping = {
        'S3': 'S3', 
        'S4': 'S4'
    }
    
    # Get last historical point for connection
    last_hist_date = None
    last_hist_pd = None
    if historical_data is not None and not historical_data.empty:
        last_hist_date = historical_data['yyyyqq'].iloc[-1]
        last_hist_pd = historical_data['historical_pd'].iloc[-1]
    
    # Collect all PD values for consistent y-axis scaling (only S3 and S4)
    all_pd_values = []
    
    # Add historical PD values
    if historical_data is not None and not historical_data.empty:
        all_pd_values.extend(historical_data['historical_pd'].dropna().tolist())
    
    # Add OVS PD values for S3 and S4 scenarios
    for ovs_model, ovs_data in ovs_forecasts.items():
        if ovs_data is not None:
            for scenario in ['S3', 'S4']:
                ovs_col = f'{ovs_model}_PD_{scenario}'
                if ovs_col in ovs_data.columns:
                    ovs_values = ovs_data[ovs_col].dropna()
                    if not ovs_values.empty:
                        all_pd_values.extend(ovs_values.tolist())
    
    # Add GCorr PD values for S3 and S4 scenarios
    gcorr_data = gcorr_forecasts.get(gcorr_type)
    if gcorr_data is not None:
        for scenario in ['S3', 'S4']:
            gcorr_scenario = scenario_mapping[scenario]
            gcorr_col = f'gcorr_{gcorr_type}_PD_{gcorr_scenario}'
            if gcorr_col in gcorr_data.columns:
                gcorr_values = gcorr_data[gcorr_col].dropna()
                if not gcorr_values.empty:
                    all_pd_values.extend(gcorr_values.tolist())
    
    # Calculate consistent y-axis limits
    if all_pd_values:
        y_min, y_max = min(all_pd_values), max(all_pd_values)
        
        # Add 10% padding
        y_range = y_max - y_min
        y_padding = max(y_range * 0.1, y_max * 0.05)  # At least 5% of max value
        y_min_plot = max(0, y_min - y_padding)  # PD can't be negative
        y_max_plot = y_max + y_padding
        
        print(f"  {country} ({gcorr_type}) PD range: {y_min:.6f} to {y_max:.6f}, using plot range: {y_min_plot:.6f} to {y_max_plot:.6f}")
    else:
        y_min_plot, y_max_plot = 0, 0.01  # Default range
    
    # Plot S3 scenario comparison
    ax1 = axes[0]
    plot_scenario_comparison(ax1, 'S3', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, 
                           ovs_colors, gcorr_color, scenario_mapping, last_hist_date, last_hist_pd, 
                           y_min_plot, y_max_plot, final_selected_rank)
    
    # Plot S4 scenario comparison
    ax2 = axes[1]
    plot_scenario_comparison(ax2, 'S4', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, 
                           ovs_colors, gcorr_color, scenario_mapping, last_hist_date, last_hist_pd, 
                           y_min_plot, y_max_plot, final_selected_rank)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08)
    
    # Save plot
    plt.savefig(PLOTS_DIR / f'ovs_top10_gcorr_{country}_{gcorr_type}_{GCORR_FORECAST_QUARTERS}q.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib rcParams to defaults
    plt.rcdefaults()
    
    print(f"  Saved top 10 comparison plot for {country} ({gcorr_type})")

def plot_scenario_comparison(ax, scenario, gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, 
                           ovs_colors, gcorr_color, scenario_mapping, last_hist_date, last_hist_pd, 
                           y_min_plot, y_max_plot, final_selected_rank=None):
    """Plot comparison for a specific scenario with all OVS models and GCorr"""
    
    # Define cutoff date - only show forecasts up to 2040
    cutoff_date = pd.Timestamp('2040-12-31')
    
    # Plot historical data (convert to percentage)
    if historical_data is not None and not historical_data.empty:
        ax.plot(historical_data['yyyyqq'], historical_data['historical_pd'] * 100, 
                color='black', linewidth=2, label='Historical', alpha=0.7)
    
    # Plot top 10 OVS forecasts
    ovs_model_names = sorted(ovs_forecasts.keys(), key=lambda x: int(x.replace('ovs_rank', '')))
    
    for i, ovs_model in enumerate(ovs_model_names):
        ovs_data = ovs_forecasts[ovs_model]
        if ovs_data is None:
            continue
            
        ovs_col = f'{ovs_model}_PD_{scenario}'
        if ovs_col in ovs_data.columns:
            ovs_forecast = ovs_data[ovs_data[ovs_col].notna()]
            # Filter to only show forecasts up to 2040
            ovs_forecast = ovs_forecast[ovs_forecast['yyyyqq'] <= cutoff_date]
            if not ovs_forecast.empty:
                # Connect to historical if available (convert to percentage)
                if last_hist_date is not None and last_hist_pd is not None:
                    plot_dates = [last_hist_date] + ovs_forecast['yyyyqq'].tolist()
                    plot_values = [last_hist_pd * 100] + (ovs_forecast[ovs_col] * 100).tolist()
                else:
                    plot_dates = ovs_forecast['yyyyqq']
                    plot_values = ovs_forecast[ovs_col] * 100
                
                rank_num = ovs_model.replace('ovs_rank', '')
                # Make final selected model bolder and more prominent
                if final_selected_rank is not None and int(rank_num) == final_selected_rank:
                    linewidth = 3.5  # Bolder line for final selected model
                    alpha = 1.0      # More opaque for final selected model
                    label_suffix = ' (SELECTED)'  # Add suffix to indicate selected model
                else:
                    linewidth = 1.5  # Standard line width for other models
                    alpha = 0.8      # Standard transparency for other models
                    label_suffix = ''
                
                ax.plot(plot_dates, plot_values, 
                        color=ovs_colors[i], linestyle='-', linewidth=linewidth, alpha=alpha,
                        label=f'OVS Rank {rank_num}{label_suffix}')
    
    # GCorr forecast
    gcorr_data = gcorr_forecasts.get(gcorr_type)
    if gcorr_data is not None:
        gcorr_scenario = scenario_mapping[scenario]
        gcorr_col = f'gcorr_{gcorr_type}_PD_{gcorr_scenario}'
        if gcorr_col in gcorr_data.columns:
            gcorr_forecast = gcorr_data[gcorr_data[gcorr_col].notna()]
            # Filter to only show forecasts up to 2040
            gcorr_forecast = gcorr_forecast[gcorr_forecast['yyyyqq'] <= cutoff_date]
            if not gcorr_forecast.empty:
                # Connect to historical if available (convert to percentage)
                if last_hist_date is not None and last_hist_pd is not None:
                    plot_dates = [last_hist_date] + gcorr_forecast['yyyyqq'].tolist()
                    plot_values = [last_hist_pd * 100] + (gcorr_forecast[gcorr_col] * 100).tolist()
                else:
                    plot_dates = gcorr_forecast['yyyyqq']
                    plot_values = gcorr_forecast[gcorr_col] * 100
                
                ax.plot(plot_dates, plot_values, 
                        color=gcorr_color, linestyle='--', linewidth=3, alpha=0.9,
                        label='GCorr (Smoothed)')
    
    # Formatting with larger fonts
    ax.axvline(pd.to_datetime(FORECAST_START), color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.set_title(f'{scenario} Scenario', fontsize=22, fontweight='bold')
    ax.set_ylabel('PD (%)', fontsize=22, fontweight='bold')
    # Remove x-axis title as requested
    ax.legend(fontsize=14, loc='best', ncol=2)  # Use smaller font and 2 columns for many models
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_ylim(y_min_plot * 100, y_max_plot * 100)  # Apply consistent y-axis limits (convert to percentage)
    ax.set_xlim(left=None, right=cutoff_date)  # Limit x-axis to 2040
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

def calculate_forecast_differences(ovs_forecasts, gcorr_forecasts):
    """Calculate differences between top 10 OVS models and GCorr forecasts"""
    differences = {}
    
    scenario_mapping = {
        'S3': 'S3',
        'S4': 'S4'
    }
    
    # Compare each OVS model with GCorr smoothed
    gcorr_data = gcorr_forecasts.get('smoothed')
    if gcorr_data is None:
        return differences
    
    for ovs_model, ovs_data in ovs_forecasts.items():
        if ovs_data is None:
            continue
        
        for scenario in ['S3', 'S4']:
            ovs_col = f'{ovs_model}_PD_{scenario}'
            gcorr_scenario = scenario_mapping[scenario]
            gcorr_col = f'gcorr_smoothed_PD_{gcorr_scenario}'
            
            if (ovs_col in ovs_data.columns and gcorr_col in gcorr_data.columns):
                
                # Merge data on common dates
                merged = pd.merge(ovs_data[['yyyyqq', ovs_col]], 
                                gcorr_data[['yyyyqq', gcorr_col]], 
                                on='yyyyqq', how='inner')
                
                if not merged.empty:
                    # Calculate differences
                    merged['diff'] = merged[ovs_col] - merged[gcorr_col]
                    merged['rel_diff'] = merged['diff'] / merged[gcorr_col] * 100
                    
                    key = f'{ovs_model}_vs_gcorr_smoothed_{scenario}'
                    differences[key] = {
                        'mean_diff': merged['diff'].mean(),
                        'mean_rel_diff': merged['rel_diff'].mean(),
                        'max_diff': merged['diff'].max(),
                        'min_diff': merged['diff'].min(),
                        'rmse': np.sqrt(np.mean(merged['diff']**2)),
                        'n_points': len(merged)
                    }
    
    return differences

def create_summary_table(all_results):
    """Create summary table of all countries"""
    summary_data = []
    
    for country, result in all_results.items():
        summary_record = {
            'country': country,
            'has_historical': result['has_historical'],
            'has_ovs': result['has_ovs'],
            'has_gcorr': result['has_gcorr'],
            'gcorr_rsq': result['metadata'].get('gcorr_rsq', np.nan),
            'gcorr_num_mv': result['metadata'].get('gcorr_num_mv', np.nan),
            'ovs_num_models': result['metadata'].get('ovs_num_models', 0),
            'ovs_best_adj_r2': result['metadata'].get('ovs_best_adj_r2', np.nan),
            'ovs_worst_adj_r2': result['metadata'].get('ovs_worst_adj_r2', np.nan)
        }
        
        # Add forecast differences for top models
        if result['differences']:
            for comparison_key, diff_stats in result['differences'].items():
                summary_record[f'{comparison_key}_mean_diff'] = diff_stats['mean_diff']
                summary_record[f'{comparison_key}_rmse'] = diff_stats['rmse']
                summary_record[f'{comparison_key}_n_points'] = diff_stats['n_points']
        
        summary_data.append(summary_record)
    
    return pd.DataFrame(summary_data)

def main():
    print(f"All OVS Models vs GCorr Forecast Comparison (S3 & S4 Scenarios Only)")
    print(f"GCorr Forecast Quarters: {GCORR_FORECAST_QUARTERS}Q")
    print(f"OVS Models: All filtered models (GCorr {GCORR_FORECAST_QUARTERS}Q Smoothed)")
    print(f"GCorr Models: {list(GCORR_TYPES.keys())}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Check if required files exist
    print("Checking required files...")
    if not Path(HISTORICAL_DATA_FILE).exists():
        print(f"ERROR: Historical data file not found: {HISTORICAL_DATA_FILE}")
        return
    if not OVS_FORECAST_DIR.exists():
        print(f"ERROR: OVS forecast directory not found: {OVS_FORECAST_DIR}")
        return  
    if not GCORR_FORECAST_DIR.exists():
        print(f"ERROR: GCorr forecast directory not found: {GCORR_FORECAST_DIR}")
        return
    print("All required files and directories found.")
    
    # Load historical data
    historical_data = load_historical_data()
    
    # Get list of countries from both OVS and GCorr directories
    ovs_countries = set()
    gcorr_countries = set()
    
    # Get OVS countries from step 8.1 output directories
    if OVS_FORECAST_DIR.exists():
        for country_dir in OVS_FORECAST_DIR.iterdir():
            if country_dir.is_dir():
                # Check if this directory has individual model forecast files
                model_files = list(country_dir.glob(f'forecast_model_rank_*_advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags.csv'))
                if model_files:
                    ovs_countries.add(country_dir.name)
    
    # Get GCorr countries (check for smoothed only)
    if GCORR_FORECAST_DIR.exists():
        for file in GCORR_FORECAST_DIR.glob('*_annualized_smoothed_pd.csv'):
            country = file.stem.replace('_annualized_smoothed_pd', '')
            gcorr_countries.add(country)
    
    # Find common countries
    common_countries = ovs_countries.intersection(gcorr_countries)
    all_countries = ovs_countries.union(gcorr_countries)
    
    print(f"OVS countries: {len(ovs_countries)}")
    print(f"GCorr countries: {len(gcorr_countries)}")
    print(f"Common countries: {len(common_countries)}")
    print(f"Total countries: {len(all_countries)}")
    
    # Process each country
    all_results = {}
    successful_comparisons = 0
    
    countries_to_process = sorted(all_countries)
    if TEST_LIMIT is not None:
        countries_to_process = countries_to_process[:TEST_LIMIT]
        print(f"Testing mode: Processing only first {TEST_LIMIT} countries")
    
    for i, country in enumerate(countries_to_process, 1):
        print(f"\n[{i}/{len(countries_to_process)}] Processing {country}...")
        
        try:
            # Initialize result structure
            result = {
                'has_historical': False,
                'has_ovs': False,
                'has_gcorr': False,
                'metadata': {},
                'differences': {}
            }
            
            # Load data
            historical_pd = get_historical_pd_for_country(historical_data, country)
            ovs_forecasts = load_ovs_all_forecasts(country)
            gcorr_forecasts = load_gcorr_forecasts(country)
            metadata = get_country_metadata(country)
            
            # Update result flags
            result['has_historical'] = historical_pd is not None
            result['has_ovs'] = len(ovs_forecasts) > 0
            result['has_gcorr'] = any(data is not None for data in gcorr_forecasts.values())
            result['metadata'] = metadata
            
            # Calculate differences if both forecasts available
            if result['has_ovs'] and result['has_gcorr']:
                differences = calculate_forecast_differences(ovs_forecasts, gcorr_forecasts)
                result['differences'] = differences
                
                # Create comparison plots
                create_comparison_plots(country, historical_pd, ovs_forecasts, gcorr_forecasts, metadata)
                successful_comparisons += 1
                
            elif result['has_ovs']:
                print(f"  Only OVS forecast available for {country}")
            elif result['has_gcorr']:
                print(f"  Only GCorr forecast available for {country}")
            else:
                print(f"  No forecasts available for {country}")
            
            all_results[country] = result
            
        except Exception as e:
            print(f"  Error processing {country}: {e}")
            continue
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    summary_df.to_csv(OUTPUT_DIR / f'ovs_top10_gcorr_comparison_summary_{GCORR_FORECAST_QUARTERS}q.csv', index=False)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("ALL OVS MODELS COMPARISON COMPLETE!")
    print(f"Successfully created comparison plots: {successful_comparisons}")
    print(f"Common countries with both forecasts: {len(common_countries)}")
    print(f"Countries with only OVS: {len(ovs_countries - gcorr_countries)}")
    print(f"Countries with only GCorr: {len(gcorr_countries - ovs_countries)}")
    
    # Summary statistics
    if not summary_df.empty:
        both_models = summary_df[summary_df['has_ovs'] & summary_df['has_gcorr']]
        
        if not both_models.empty:
            print(f"\nModel Performance Comparison (countries with both models):")
            print(f"Average number of OVS models per country: {both_models['ovs_num_models'].mean():.1f}")
            print(f"Average best OVS Adj R²: {both_models['ovs_best_adj_r2'].mean():.3f}")
            print(f"Average worst OVS Adj R²: {both_models['ovs_worst_adj_r2'].mean():.3f}")
            print(f"Average GCorr R²: {both_models['gcorr_rsq'].mean():.3f}")
    
    print(f"\nOutput files:")
    print(f"- Comparison plots: {PLOTS_DIR}")
    print(f"- Summary table: ovs_top10_gcorr_comparison_summary_{GCORR_FORECAST_QUARTERS}q.csv")
    print(f"- Focus: S3 and S4 scenarios only")
    print(f"- Models: All filtered OVS (GCorr {GCORR_FORECAST_QUARTERS}Q smoothed) vs GCorr (smoothed)")
    print(f"- Emphasis: Final selected model from step 7 is highlighted")
    print(f"- Configuration: {GCORR_FORECAST_QUARTERS} quarter GCorr forecast horizon")

if __name__ == "__main__":
    main()
