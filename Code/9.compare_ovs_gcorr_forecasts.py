import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
HISTORICAL_DATA_FILE = 'Output/4.transformation/transformed_data.csv'
OVS_FORECAST_DIR = Path('Output/8.scenario_forecast')
GCORR_FORECAST_DIR = Path('gcorr-research-delivery-validation/Output/gcorr_scenario_plots/annualized_data')
FORECAST_START = '2025-07-01'

# Output directories
OUTPUT_DIR = Path('Output/9.ovs_gcorr_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model combinations to use from OVS
OVS_COMBINATIONS = {
    'ovs_historical': 'advanced_no_lags',  # OVS trained on historical data only
    'ovs_gcorr_unsmoothed': 'advanced_gcorr40q_no_lags',  # OVS trained on GCorr unsmoothed data
    'ovs_gcorr_smoothed': 'advanced_gcorr40q_smoothed_no_lags'  # OVS trained on GCorr smoothed data
}

# GCorr data types to compare
GCORR_TYPES = {
    'unsmoothed': 'unsmoothed',
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

def load_ovs_forecasts(country):
    """Load all OVS forecast data for a specific country"""
    ovs_forecasts = {}
    
    for ovs_type, combination in OVS_COMBINATIONS.items():
        ovs_file = OVS_FORECAST_DIR / country / f'forecast_all_scenarios_{combination}.csv'
        
        if not ovs_file.exists():
            print(f"  OVS forecast file not found for {country} ({ovs_type}): {ovs_file.name}")
            ovs_forecasts[ovs_type] = None
            continue
        
        try:
            ovs_data = pd.read_csv(ovs_file)
            ovs_data['yyyyqq'] = pd.to_datetime(ovs_data['yyyyqq'])
            
            # Rename columns to distinguish from GCorr and other OVS types
            ovs_renamed = ovs_data.rename(columns={
                'predicted_dlnPD_Baseline': f'{ovs_type}_dlnPD_Baseline',
                'predicted_PD_Baseline': f'{ovs_type}_PD_Baseline',
                'predicted_dlnPD_S1': f'{ovs_type}_dlnPD_S1',
                'predicted_PD_S1': f'{ovs_type}_PD_S1',
                'predicted_dlnPD_S3': f'{ovs_type}_dlnPD_S3',
                'predicted_PD_S3': f'{ovs_type}_PD_S3',
                'predicted_dlnPD_S4': f'{ovs_type}_dlnPD_S4',
                'predicted_PD_S4': f'{ovs_type}_PD_S4'
            })
            
            ovs_forecasts[ovs_type] = ovs_renamed
            
        except Exception as e:
            print(f"  Error loading OVS forecast for {country} ({ovs_type}): {e}")
            ovs_forecasts[ovs_type] = None
    
    return ovs_forecasts

def load_gcorr_forecasts(country):
    """Load both smoothed and unsmoothed GCorr forecast data for a specific country"""
    gcorr_forecasts = {}
    
    for gcorr_type, file_type in GCORR_TYPES.items():
        gcorr_file = GCORR_FORECAST_DIR / f'{country}_annualized_{file_type}_pd.csv'
        
        if not gcorr_file.exists():
            print(f"  GCorr forecast file not found for {country} ({gcorr_type}): {gcorr_file.name}")
            gcorr_forecasts[gcorr_type] = None
            continue
        
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
            
            # Rename columns to distinguish from OVS and other GCorr types
            scenario_columns = [col for col in gcorr_pivot.columns if col != 'date']
            rename_dict = {col: f'gcorr_{gcorr_type}_PD_{col}' for col in scenario_columns}
            gcorr_pivot = gcorr_pivot.rename(columns=rename_dict)
            gcorr_pivot = gcorr_pivot.rename(columns={'date': 'yyyyqq'})
            
            gcorr_forecasts[gcorr_type] = gcorr_pivot
            
        except Exception as e:
            print(f"  Error loading GCorr forecast for {country} ({gcorr_type}): {e}")
            gcorr_forecasts[gcorr_type] = None
    
    return gcorr_forecasts

def get_country_metadata(country):
    """Get metadata about the country from both models"""
    metadata = {'country': country}
    
    # Try to get OVS metadata for all combination types
    try:
        ovs_summary_file = OVS_FORECAST_DIR / country / f'{country}_all_combinations_summary.csv'
        if ovs_summary_file.exists():
            ovs_summary = pd.read_csv(ovs_summary_file)
            
            # Get metadata for each OVS combination type
            for ovs_type, combination in OVS_COMBINATIONS.items():
                ovs_filtered = ovs_summary[ovs_summary['combination'] == combination]
                if not ovs_filtered.empty:
                    metadata[f'{ovs_type}_adj_r2'] = ovs_filtered['model_adj_r2'].tolist()[0]
                    metadata[f'{ovs_type}_pd_mape'] = ovs_filtered['historical_pd_mape'].tolist()[0]
                    metadata[f'{ovs_type}_pd_correlation'] = ovs_filtered['historical_pd_correlation'].tolist()[0]
    except Exception as e:
        print(f"  Warning: Could not load OVS metadata for {country}: {e}")
    
    # Try to get detailed OVS variable information from filtered results
    try:
        ovs_combinations_map = {
            'ovs_historical': 'advanced',
            'ovs_gcorr_unsmoothed': 'advanced_gcorr40q',
            'ovs_gcorr_smoothed': 'advanced_gcorr40q_smoothed'
        }
        
        for ovs_type, file_suffix in ovs_combinations_map.items():
            ovs_detailed_file = Path(f'Output/7.filtered_ovs_results/filtered_ovs_results_{file_suffix}.csv')
            if ovs_detailed_file.exists():
                ovs_detailed = pd.read_csv(ovs_detailed_file)
                country_results = ovs_detailed[ovs_detailed['country'] == country]
                if not country_results.empty:
                    # Get the first (best) model for this country
                    best_model = country_results.iloc[0]
                    
                    # Extract MV variables and coefficients
                    mv_info = []
                    for i in range(1, 5):  # MV1 to MV4
                        mv_name = best_model.get(f'MV{i}')
                        mv_coef = best_model.get(f'MV{i}_coefficient')
                        if pd.notna(mv_name) and pd.notna(mv_coef):
                            mv_info.append(f"{mv_name}: {mv_coef:.3f}")
                    
                    metadata[f'{ovs_type}_mv_info'] = '; '.join(mv_info) if mv_info else 'No MV variables'
    except Exception as e:
        print(f"  Warning: Could not load detailed OVS variable info for {country}: {e}")
    
    # Try to get GCorr metadata
    try:
        gcorr_summary_file = Path('gcorr-research-delivery-validation/Output/gcorr_scenario_plots/gcorr_scenario_forecast_summary_annualized.csv')
        if gcorr_summary_file.exists():
            gcorr_summary = pd.read_csv(gcorr_summary_file)
            gcorr_country = gcorr_summary[gcorr_summary['country'] == country]
            if not gcorr_country.empty:
                metadata['gcorr_rsq'] = gcorr_country['rsq'].iloc[0]
                metadata['gcorr_mv_variables'] = gcorr_country['mv_variables'].iloc[0]
                metadata['gcorr_num_mv'] = gcorr_country['num_mv'].iloc[0]
    except Exception as e:
        print(f"  Warning: Could not load GCorr metadata for {country}: {e}")
    
    # Try to get detailed GCorr variable information
    try:
        gcorr_detailed_file = Path('gcorr-research-delivery-validation/Output/gcorr_scenario_plots/gcorr_scenario_forecast_summary_annualized.csv')
        if gcorr_detailed_file.exists():
            gcorr_detailed = pd.read_csv(gcorr_detailed_file)
            country_results = gcorr_detailed[gcorr_detailed['country'] == country]
            if not country_results.empty:
                # Get the first (and likely only) model for this country
                best_model = country_results.iloc[0]
                
                # Extract MV variables and coefficients
                mv_info = []
                for i in range(1, 5):  # mv1 to mv4
                    mv_name = best_model.get(f'mv{i}')
                    mv_coef = best_model.get(f'beta{i}')
                    if pd.notna(mv_name) and pd.notna(mv_coef):
                        mv_info.append(f"{mv_name}: {mv_coef:.3f}")
                
                metadata['gcorr_mv_info'] = '; '.join(mv_info) if mv_info else 'No MV variables'
    except Exception as e:
        print(f"  Warning: Could not load detailed GCorr variable info for {country}: {e}")
    
    return metadata

def create_comparison_plots(country, historical_data, ovs_forecasts, gcorr_forecasts, metadata):
    """Create comparison plots for a country - both smoothed and unsmoothed"""
    
    # Create unsmoothed comparison
    if gcorr_forecasts.get('unsmoothed') is not None:
        create_single_comparison_plot(country, 'unsmoothed', historical_data, ovs_forecasts, gcorr_forecasts, metadata)
    
    # Create smoothed comparison
    if gcorr_forecasts.get('smoothed') is not None:
        create_single_comparison_plot(country, 'smoothed', historical_data, ovs_forecasts, gcorr_forecasts, metadata)

def create_single_comparison_plot(country, gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, metadata):
    """Create comparison plot for a country with consistent scaling for specific GCorr type"""
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'OVS vs GCorr Forecast Comparison: {country} ({gcorr_type.capitalize()})', fontsize=16, y=0.98)
    
    # Define colors for scenarios
    colors = {
        'Baseline': 'green',
        'S1': 'blue', 
        'S3': 'orange',
        'S4': 'red'
    }
    
    # Define scenario mappings (GCorr uses 'baseline' while OVS uses 'Baseline')
    scenario_mapping = {
        'Baseline': 'baseline',
        'S1': 'S1',
        'S3': 'S3', 
        'S4': 'S4'
    }
    
    # Get last historical point for connection
    last_hist_date = None
    last_hist_pd = None
    if historical_data is not None and not historical_data.empty:
        last_hist_date = historical_data['yyyyqq'].iloc[-1]
        last_hist_pd = historical_data['historical_pd'].iloc[-1]
    
    # Collect all PD values for consistent y-axis scaling
    all_pd_values = []
    
    # Add historical PD values
    if historical_data is not None and not historical_data.empty:
        all_pd_values.extend(historical_data['historical_pd'].dropna().tolist())
    
    # Add OVS PD values for all scenarios and all OVS types
    for ovs_type, ovs_data in ovs_forecasts.items():
        if ovs_data is not None:
            for scenario in ['Baseline', 'S1', 'S3', 'S4']:
                ovs_col = f'{ovs_type}_PD_{scenario}'
                if ovs_col in ovs_data.columns:
                    ovs_values = ovs_data[ovs_col].dropna()
                    if not ovs_values.empty:
                        all_pd_values.extend(ovs_values.tolist())
    
    # Add GCorr PD values for all scenarios
    gcorr_data = gcorr_forecasts.get(gcorr_type)
    if gcorr_data is not None:
        for scenario in ['Baseline', 'S1', 'S3', 'S4']:
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
    
    # Plot 1: Baseline scenario comparison
    ax1 = axes[0, 0]
    plot_scenario_comparison(ax1, 'Baseline', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Plot 2: S1 scenario comparison
    ax2 = axes[0, 1]
    plot_scenario_comparison(ax2, 'S1', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Plot 3: S3 scenario comparison
    ax3 = axes[1, 0]
    plot_scenario_comparison(ax3, 'S3', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Plot 4: S4 scenario comparison
    ax4 = axes[1, 1]
    plot_scenario_comparison(ax4, 'S4', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Add metadata text with MV variable information
    hist_text = f"OVS (Historical): Adj R² = {metadata.get('ovs_historical_adj_r2', 'N/A'):.3f}, " \
                f"PD MAPE = {metadata.get('ovs_historical_pd_mape', 'N/A'):.1f}%\n" \
                f"  MV Variables: {metadata.get('ovs_historical_mv_info', 'N/A')}\n"
    
    # Add metadata for GCorr-trained OVS models
    if gcorr_type == 'unsmoothed':
        gcorr_ovs_text = f"OVS (GCorr Unsmoothed): Adj R² = {metadata.get('ovs_gcorr_unsmoothed_adj_r2', 'N/A'):.3f}, " \
                        f"PD MAPE = {metadata.get('ovs_gcorr_unsmoothed_pd_mape', 'N/A'):.1f}%\n" \
                        f"  MV Variables: {metadata.get('ovs_gcorr_unsmoothed_mv_info', 'N/A')}\n"
    else:
        gcorr_ovs_text = f"OVS (GCorr Smoothed): Adj R² = {metadata.get('ovs_gcorr_smoothed_adj_r2', 'N/A'):.3f}, " \
                        f"PD MAPE = {metadata.get('ovs_gcorr_smoothed_pd_mape', 'N/A'):.1f}%\n" \
                        f"  MV Variables: {metadata.get('ovs_gcorr_smoothed_mv_info', 'N/A')}\n"
    
    gcorr_text = f"GCorr: R² = {metadata.get('gcorr_rsq', 'N/A'):.3f}, " \
                f"MV = {metadata.get('gcorr_mv_variables', 'N/A')}"
    
    metadata_text = hist_text + gcorr_ovs_text + gcorr_text
    
    fig.text(0.02, 0.02, metadata_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    
    # Save plot
    plt.savefig(PLOTS_DIR / f'ovs_gcorr_{country}_{gcorr_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot for {country} ({gcorr_type})")

def plot_scenario_comparison(ax, scenario, gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot):
    """Plot comparison for a specific scenario with consistent y-axis scaling"""
    
    # Plot historical data
    if historical_data is not None and not historical_data.empty:
        ax.plot(historical_data['yyyyqq'], historical_data['historical_pd'], 
                'black', linewidth=2, label='Historical', alpha=0.7)
    
    color = colors[scenario]
    
    # Define line styles for different OVS types (consistent across both chart types)
    ovs_styles = {
        'ovs_historical': '-',
        'ovs_gcorr_unsmoothed': ':',
        'ovs_gcorr_smoothed': ':'
    }
    
    # Define labels for different OVS types
    ovs_labels = {
        'ovs_historical': 'OVS (Historical)',
        'ovs_gcorr_unsmoothed': 'OVS (GCorr Unsmoothed)',
        'ovs_gcorr_smoothed': 'OVS (GCorr Smoothed)'
    }
    
    # Plot OVS forecasts - prioritize the matching GCorr type
    for ovs_type, ovs_data in ovs_forecasts.items():
        if ovs_data is None:
            continue
            
        # Skip non-matching types for cleaner plots
        if gcorr_type == 'unsmoothed' and ovs_type == 'ovs_gcorr_smoothed':
            continue
        if gcorr_type == 'smoothed' and ovs_type == 'ovs_gcorr_unsmoothed':
            continue
            
        ovs_col = f'{ovs_type}_PD_{scenario}'
        if ovs_col in ovs_data.columns:
            ovs_forecast = ovs_data[ovs_data[ovs_col].notna()]
            if not ovs_forecast.empty:
                # Connect to historical if available
                if last_hist_date is not None and last_hist_pd is not None:
                    plot_dates = [last_hist_date] + ovs_forecast['yyyyqq'].tolist()
                    plot_values = [last_hist_pd] + ovs_forecast[ovs_col].tolist()
                else:
                    plot_dates = ovs_forecast['yyyyqq']
                    plot_values = ovs_forecast[ovs_col]
                
                ax.plot(plot_dates, plot_values, 
                        color=color, linestyle=ovs_styles[ovs_type], linewidth=2.5, alpha=0.9,
                        label=ovs_labels[ovs_type])
    
    # GCorr forecast
    gcorr_data = gcorr_forecasts.get(gcorr_type)
    if gcorr_data is not None:
        gcorr_scenario = scenario_mapping[scenario]
        gcorr_col = f'gcorr_{gcorr_type}_PD_{gcorr_scenario}'
        if gcorr_col in gcorr_data.columns:
            gcorr_forecast = gcorr_data[gcorr_data[gcorr_col].notna()]
            if not gcorr_forecast.empty:
                # Connect to historical if available
                if last_hist_date is not None and last_hist_pd is not None:
                    plot_dates = [last_hist_date] + gcorr_forecast['yyyyqq'].tolist()
                    plot_values = [last_hist_pd] + gcorr_forecast[gcorr_col].tolist()
                else:
                    plot_dates = gcorr_forecast['yyyyqq']
                    plot_values = gcorr_forecast[gcorr_col]
                
                ax.plot(plot_dates, plot_values, 
                        color=color, linestyle='--', linewidth=3, alpha=0.9,
                        label=f'GCorr ({gcorr_type.capitalize()})')
    
    # Formatting
    ax.axvline(pd.to_datetime(FORECAST_START), color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'{scenario} Scenario')
    ax.set_ylabel('PD')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(y_min_plot, y_max_plot)  # Apply consistent y-axis limits

def calculate_forecast_differences(ovs_forecasts, gcorr_forecasts):
    """Calculate differences between OVS and GCorr forecasts"""
    differences = {}
    
    scenario_mapping = {
        'Baseline': 'baseline',
        'S1': 'S1',
        'S3': 'S3',
        'S4': 'S4'
    }
    
    # Compare each OVS type with corresponding GCorr type
    for ovs_type, ovs_data in ovs_forecasts.items():
        if ovs_data is None:
            continue
            
        # Determine which GCorr type to compare with
        if ovs_type == 'ovs_gcorr_smoothed':
            gcorr_type = 'smoothed'
        else:
            gcorr_type = 'unsmoothed'  # Default for historical and unsmoothed
            
        gcorr_data = gcorr_forecasts.get(gcorr_type)
        if gcorr_data is None:
            continue
        
        for scenario in ['Baseline', 'S1', 'S3', 'S4']:
            ovs_col = f'{ovs_type}_PD_{scenario}'
            gcorr_scenario = scenario_mapping[scenario]
            gcorr_col = f'gcorr_{gcorr_type}_PD_{gcorr_scenario}'
            
            if (ovs_col in ovs_data.columns and gcorr_col in gcorr_data.columns):
                
                # Merge data on common dates
                merged = pd.merge(ovs_data[['yyyyqq', ovs_col]], 
                                gcorr_data[['yyyyqq', gcorr_col]], 
                                on='yyyyqq', how='inner')
                
                if not merged.empty:
                    # Calculate differences
                    merged['diff'] = merged[ovs_col] - merged[gcorr_col]
                    merged['rel_diff'] = merged['diff'] / merged[gcorr_col] * 100
                    
                    key = f'{ovs_type}_vs_{gcorr_type}_{scenario}'
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
            'gcorr_num_mv': result['metadata'].get('gcorr_num_mv', np.nan)
        }
        
        # Add metadata for all OVS types
        for ovs_type in ['ovs_historical', 'ovs_gcorr_unsmoothed', 'ovs_gcorr_smoothed']:
            summary_record[f'{ovs_type}_adj_r2'] = result['metadata'].get(f'{ovs_type}_adj_r2', np.nan)
            summary_record[f'{ovs_type}_pd_mape'] = result['metadata'].get(f'{ovs_type}_pd_mape', np.nan)
        
        # Add forecast differences
        if result['differences']:
            for comparison_key, diff_stats in result['differences'].items():
                summary_record[f'{comparison_key}_mean_diff'] = diff_stats['mean_diff']
                summary_record[f'{comparison_key}_rmse'] = diff_stats['rmse']
                summary_record[f'{comparison_key}_n_points'] = diff_stats['n_points']
        
        summary_data.append(summary_record)
    
    return pd.DataFrame(summary_data)

def main():
    print("OVS vs GCorr Forecast Comparison")
    print(f"OVS Models: {list(OVS_COMBINATIONS.keys())}")
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
    
    # Get OVS countries (check for any of the combinations)
    if OVS_FORECAST_DIR.exists():
        for country_dir in OVS_FORECAST_DIR.iterdir():
            if country_dir.is_dir():
                # Check if at least one of the combinations exists
                for combination in OVS_COMBINATIONS.values():
                    forecast_file = country_dir / f'forecast_all_scenarios_{combination}.csv'
                    if forecast_file.exists():
                        ovs_countries.add(country_dir.name)
                        break
    
    # Get GCorr countries (check for both smoothed and unsmoothed)
    if GCORR_FORECAST_DIR.exists():
        for file_type in GCORR_TYPES.values():
            for file in GCORR_FORECAST_DIR.glob(f'*_annualized_{file_type}_pd.csv'):
                country = file.stem.replace(f'_annualized_{file_type}_pd', '')
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
            ovs_forecasts = load_ovs_forecasts(country)
            gcorr_forecasts = load_gcorr_forecasts(country)
            metadata = get_country_metadata(country)
            
            # Update result flags
            result['has_historical'] = historical_pd is not None
            result['has_ovs'] = any(data is not None for data in ovs_forecasts.values())
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
    summary_df.to_csv(OUTPUT_DIR / 'ovs_gcorr_comparison_summary.csv', index=False)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print(f"Successfully created comparison plots: {successful_comparisons}")
    print(f"Common countries with both forecasts: {len(common_countries)}")
    print(f"Countries with only OVS: {len(ovs_countries - gcorr_countries)}")
    print(f"Countries with only GCorr: {len(gcorr_countries - ovs_countries)}")
    
    # Summary statistics
    if not summary_df.empty:
        both_models = summary_df[summary_df['has_ovs'] & summary_df['has_gcorr']]
        
        if not both_models.empty:
            print(f"\nModel Performance Comparison (countries with both models):")
            
            # Check for historical OVS data
            if 'ovs_historical_adj_r2' in both_models.columns:
                print(f"Average OVS (Historical) Adj R²: {both_models['ovs_historical_adj_r2'].mean():.3f}")
                print(f"Average OVS (Historical) PD MAPE: {both_models['ovs_historical_pd_mape'].mean():.1f}%")
            
            # Check for GCorr-trained OVS data
            if 'ovs_gcorr_unsmoothed_adj_r2' in both_models.columns:
                print(f"Average OVS (GCorr Unsmoothed) Adj R²: {both_models['ovs_gcorr_unsmoothed_adj_r2'].mean():.3f}")
            
            if 'ovs_gcorr_smoothed_adj_r2' in both_models.columns:
                print(f"Average OVS (GCorr Smoothed) Adj R²: {both_models['ovs_gcorr_smoothed_adj_r2'].mean():.3f}")
            
            # GCorr performance
            print(f"Average GCorr R²: {both_models['gcorr_rsq'].mean():.3f}")
            
            # Forecast difference statistics
            baseline_diff_cols = [col for col in both_models.columns if 'Baseline_mean_diff' in col]
            if baseline_diff_cols:
                print(f"\nBaseline Scenario Differences (OVS - GCorr):")
                for col in baseline_diff_cols:
                    print(f"Average difference ({col}): {both_models[col].mean():.4f}")
    
    print(f"\nOutput files:")
    print(f"- Comparison plots: {PLOTS_DIR}")
    print(f"- Summary table: ovs_gcorr_comparison_summary.csv")
    print(f"- OVS models: {list(OVS_COMBINATIONS.keys())}")
    print(f"- GCorr models: {list(GCORR_TYPES.keys())}")
    
    print(f"\nNote: Both models use annual PD for proper comparison")
    print(f"- OVS forecasts are already in annual terms")
    print(f"- GCorr forecasts converted from quarterly to annual")

if __name__ == "__main__":
    main() 