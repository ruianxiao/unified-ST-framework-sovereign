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
GCORR_FORECAST_QUARTERS = 40  # Number of GCorr forecast quarters (20 or 40)

# Output directories
OUTPUT_DIR = Path('Output/9.ovs_gcorr_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model combinations to use from OVS
OVS_COMBINATIONS = {
    'ovs_historical': 'advanced_has_no_lags',  # OVS trained on historical data only
    'ovs_gcorr_unsmoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_has_no_lags',  # OVS trained on GCorr unsmoothed data
    'ovs_gcorr_smoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed_has_no_lags'  # OVS trained on GCorr smoothed data
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
    
    # Get OVS metadata from step 7 final top models files
    try:
        ovs_combinations_map = {
            'ovs_historical': 'advanced',
            'ovs_gcorr_unsmoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q',
            'ovs_gcorr_smoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed'
        }
        
        for ovs_type, file_suffix in ovs_combinations_map.items():
            ovs_detailed_file = Path(f'Output/7.filtered_ovs_results/final_top_models_{file_suffix}.csv')
            if ovs_detailed_file.exists():
                ovs_detailed = pd.read_csv(ovs_detailed_file)
                country_results = ovs_detailed[ovs_detailed['country'] == country]
                if not country_results.empty:
                    # Final top models files already contain the selected top model per country
                    best_model = country_results.iloc[0]
                    
                    # Get adjusted R² from the final top models results
                    metadata[f'{ovs_type}_adj_r2'] = best_model.get('adj_r2')
                    
                    # Extract MV variables and coefficients
                    mv_info = []
                    for i in range(1, 5):  # MV1 to MV4
                        mv_name = best_model.get(f'MV{i}')
                        mv_coef = best_model.get(f'MV{i}_coefficient')
                        if pd.notna(mv_name) and pd.notna(mv_coef):
                            mv_info.append(f"{mv_name}: {mv_coef:.3f}")
                    
                    metadata[f'{ovs_type}_mv_info'] = '; '.join(mv_info) if mv_info else 'No MV variables'
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
    
    # Set larger font sizes for all plot elements
    plt.rcParams.update({
        'font.size': 18,           # Base font size
        'axes.titlesize': 22,      # Subplot titles
        'axes.labelsize': 22,      # Axis labels (increased)
        'xtick.labelsize': 18,     # X-axis tick labels (increased)
        'ytick.labelsize': 18,     # Y-axis tick labels (increased)
        'legend.fontsize': 18,     # Legend text (increased)
        'figure.titlesize': 24     # Main title
    })
    
    # Set up the plot with larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle(f'OVS vs GCorr Forecast Comparison: {country} ({gcorr_type.capitalize()})', fontsize=24, y=0.96)
    
    # Define colors for models (consistent across all scenarios)
    model_colors = {
        'historical': 'black',
        'ovs_historical': 'blue',
        'ovs_gcorr_unsmoothed': 'green', 
        'ovs_gcorr_smoothed': 'orange',
        'gcorr_unsmoothed': 'red',
        'gcorr_smoothed': 'purple'
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
    plot_scenario_comparison(ax1, 'Baseline', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, model_colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Plot 2: S1 scenario comparison
    ax2 = axes[0, 1]
    plot_scenario_comparison(ax2, 'S1', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, model_colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Plot 3: S3 scenario comparison
    ax3 = axes[1, 0]
    plot_scenario_comparison(ax3, 'S3', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, model_colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    # Plot 4: S4 scenario comparison
    ax4 = axes[1, 1]
    plot_scenario_comparison(ax4, 'S4', gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, model_colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    # Save plot
    plt.savefig(PLOTS_DIR / f'ovs_gcorr_{country}_{gcorr_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib rcParams to defaults
    plt.rcdefaults()
    
    print(f"  Saved comparison plot for {country} ({gcorr_type})")

def plot_scenario_comparison(ax, scenario, gcorr_type, historical_data, ovs_forecasts, gcorr_forecasts, model_colors, scenario_mapping, last_hist_date, last_hist_pd, y_min_plot, y_max_plot):
    """Plot comparison for a specific scenario with consistent y-axis scaling"""
    
    # Define cutoff date - only show forecasts up to 2040
    cutoff_date = pd.Timestamp('2040-12-31')
    
    # Plot historical data (convert to percentage)
    if historical_data is not None and not historical_data.empty:
        ax.plot(historical_data['yyyyqq'], historical_data['historical_pd'] * 100, 
                color=model_colors['historical'], linewidth=2, label='Historical', alpha=0.7)
    
    # Define labels and line styles for different OVS types
    ovs_labels = {
        'ovs_historical': 'OVS (Historical)',
        'ovs_gcorr_unsmoothed': 'OVS (GCorr Unsmoothed)',
        'ovs_gcorr_smoothed': 'OVS (GCorr Smoothed)'
    }
    
    ovs_line_styles = {
        'ovs_historical': ':',  # dotted line for OVS Historical
        'ovs_gcorr_unsmoothed': '-',  # solid line for OVS with GCorr
        'ovs_gcorr_smoothed': '-'  # solid line for OVS with GCorr
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
                
                ax.plot(plot_dates, plot_values, 
                        color=model_colors[ovs_type], linestyle=ovs_line_styles[ovs_type], linewidth=2.5, alpha=0.9,
                        label=ovs_labels[ovs_type])
    
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
                
                gcorr_color_key = f'gcorr_{gcorr_type}'
                ax.plot(plot_dates, plot_values, 
                        color=model_colors[gcorr_color_key], linestyle='--', linewidth=3, alpha=0.9,
                        label=f'GCorr ({gcorr_type.capitalize()})')
    
    # Formatting with larger fonts
    ax.axvline(pd.to_datetime(FORECAST_START), color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.set_title(f'{scenario} Scenario', fontsize=22, fontweight='bold')
    ax.set_ylabel('PD (%)', fontsize=22, fontweight='bold')
    # Remove x-axis title as requested
    ax.legend(fontsize=18, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_ylim(y_min_plot * 100, y_max_plot * 100)  # Apply consistent y-axis limits (convert to percentage)
    ax.set_xlim(left=None, right=cutoff_date)  # Limit x-axis to 2040
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

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
        
        # Add forecast differences
        if result['differences']:
            for comparison_key, diff_stats in result['differences'].items():
                summary_record[f'{comparison_key}_mean_diff'] = diff_stats['mean_diff']
                summary_record[f'{comparison_key}_rmse'] = diff_stats['rmse']
                summary_record[f'{comparison_key}_n_points'] = diff_stats['n_points']
        
        summary_data.append(summary_record)
    
    return pd.DataFrame(summary_data)

def get_regional_classifications():
    """Get regional country classifications matching generate_report_tables.py"""
    return {
        'Advanced Europe': [
            'AUT', 'BEL', 'CHE', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 
            'GBR', 'IRL', 'ITA', 'LUX', 'NLD', 'NOR', 'SWE'
        ],
        'North America': ['CAN', 'USA'],
        'Asia-Pacific': [
            'AUS', 'CHN', 'HKG', 'IND', 'JPN', 'KOR', 'NZL', 'PAK', 
            'PHL', 'SGP', 'THA', 'BGD', 'LKA', 'TWN', 'VNM', 'IDN', 'MYS'
        ],
        'Latin America': ['BRA', 'CHL', 'COL', 'MEX', 'PER', 'DOM', 'JAM', 'PAN'],
        'Middle East/Africa': ['BHR', 'ISR', 'KAZ', 'NGA', 'OMN', 'QAT', 'ZAF', 'EGY', 'MAR', 'TUN'],
        'Emerging Europe': [
            'BGR', 'CYP', 'CZE', 'EST', 'GRC', 'HRV', 'HUN', 'LTU', 
            'LVA', 'MLT', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'TUR'
        ]
    }

def generate_scenario_projection_table(all_results, historical_data):
    """Generate comprehensive scenario projection table including historical crisis periods and model forecasts"""
    print("\n🔄 Generating Scenario Projection Table (Table 4.2)...")
    
    scenario_analysis = []
    
    # Define all model types we're analyzing
    model_types = {
        'OVS Historical': 'ovs_historical',
        'OVS GCorr Unsmoothed': 'ovs_gcorr_unsmoothed', 
        'OVS GCorr Smoothed': 'ovs_gcorr_smoothed',
        'GCorr Unsmoothed': 'gcorr_unsmoothed',
        'GCorr Smoothed': 'gcorr_smoothed'
    }
    
    # Define scenarios to analyze (baseline needed for calculations but not included in final table)
    scenarios_for_calculation = ['Baseline', 'S1', 'S3', 'S4']
    scenarios_for_output = ['S1', 'S3', 'S4']  # Exclude baseline from final table
    
    # Calculate non-crisis historical baseline for reference
    if historical_data is not None:
        # Filter to exclude Serbia only
        historical_data = historical_data[historical_data['cinc'] != 'SRB'].copy()
        
        # Define crisis periods to exclude from baseline calculation
        crisis_periods = [
            (pd.Timestamp('2008-01-01'), pd.Timestamp('2009-12-31')),  # GFC
            (pd.Timestamp('2010-01-01'), pd.Timestamp('2012-12-31')),  # European Crisis
            (pd.Timestamp('2020-01-01'), pd.Timestamp('2021-12-31'))   # COVID
        ]
        
        # Create mask for all crisis periods
        crisis_mask = pd.Series(False, index=historical_data.index)
        for start_date, end_date in crisis_periods:
            period_mask = (
                (historical_data['yyyyqq'] >= start_date) & 
                (historical_data['yyyyqq'] <= end_date)
            )
            crisis_mask = crisis_mask | period_mask
        
        # Calculate non-crisis baseline
        non_crisis_data = historical_data[~crisis_mask & historical_data['cdsiedf5'].notna()]
        historical_baseline_pd = non_crisis_data['cdsiedf5'].mean() if not non_crisis_data.empty else np.nan
        print(f"  📊 Historical non-crisis baseline: {historical_baseline_pd*10000:.1f} bps")
        
        # Add historical crisis analysis to the scenario table
        print("  📊 Adding historical crisis periods to analysis...")
        crisis_info = {
            'Global Financial Crisis': ('2008-2009', pd.Timestamp('2008-01-01'), pd.Timestamp('2009-12-31')),
            'European Debt Crisis': ('2010-2012', pd.Timestamp('2010-01-01'), pd.Timestamp('2012-12-31')),
            'COVID-19 Pandemic': ('2020-2021', pd.Timestamp('2020-01-01'), pd.Timestamp('2021-12-31'))
        }
        
        for crisis_name, (period_desc, start_date, end_date) in crisis_info.items():
            crisis_data = historical_data[
                (historical_data['yyyyqq'] >= start_date) & 
                (historical_data['yyyyqq'] <= end_date) &
                historical_data['cdsiedf5'].notna()
            ].copy()
            
            if not crisis_data.empty:
                # Calculate country-specific peaks (consistent with scenario methodology)
                country_peaks = []
                for country in crisis_data['cinc'].unique():
                    if country == 'SRB':  # Exclude Serbia only
                        continue
                    country_crisis_data = crisis_data[crisis_data['cinc'] == country]
                    if len(country_crisis_data) > 0:
                        country_peak = country_crisis_data['cdsiedf5'].max()
                        country_peaks.append(country_peak)
                
                if country_peaks:
                    # Calculate average of country-specific peaks
                    avg_crisis_peak = np.mean(country_peaks)
                    peak_to_baseline_ratio = avg_crisis_peak / historical_baseline_pd if historical_baseline_pd > 0 else np.nan
                    
                    scenario_analysis.append({
                        'Model_Type': f'Historical Crisis: {crisis_name}',
                        'Scenario': period_desc,
                        'Countries_Analyzed': len(country_peaks),
                        'Avg_PD_bps': round(avg_crisis_peak * 10000, 1),
                        'Stress_Ratio': round(peak_to_baseline_ratio, 2)  # Peak vs historical baseline for crisis periods
                    })
                    
                    print(f"    ✅ {crisis_name}: {avg_crisis_peak*10000:.1f} bps avg peak across {len(country_peaks)} countries")
        
    else:
        historical_baseline_pd = np.nan  # Default fallback
    
    # Process each model type
    for model_name, model_key in model_types.items():
        print(f"    📊 Processing {model_name}...")
        
        # Collect data for this model across all countries (including baseline for calculations)
        model_scenario_data = {scenario: [] for scenario in scenarios_for_calculation}
        countries_with_data = []
        
        for country, result in all_results.items():
            if country == 'SRB':  # Exclude Serbia only
                continue
                
            # Determine data source based on model type
            if model_key.startswith('ovs_'):
                # OVS model data
                if not result['has_ovs']:
                    continue
                    
                # Load OVS forecast data for this country
                ovs_forecasts = load_ovs_forecasts(country)
                model_data = ovs_forecasts.get(model_key)
                
                if model_data is not None:
                    countries_with_data.append(country)
                    
                    for scenario in scenarios_for_calculation:
                        pd_col = f'{model_key}_PD_{scenario}'
                        if pd_col in model_data.columns:
                            scenario_values = model_data[pd_col].dropna()
                            if not scenario_values.empty:
                                # Use country-specific peak (max value) for scenario
                                country_peak = scenario_values.max()
                                model_scenario_data[scenario].append(country_peak)
                            
            elif model_key.startswith('gcorr_'):
                # GCorr model data
                if not result['has_gcorr']:
                    continue
                    
                gcorr_type = model_key.replace('gcorr_', '')  # 'unsmoothed' or 'smoothed'
                gcorr_forecasts = load_gcorr_forecasts(country)
                model_data = gcorr_forecasts.get(gcorr_type)
                
                if model_data is not None:
                    countries_with_data.append(country)
                    
                    scenario_mapping = {
                        'Baseline': 'baseline',
                        'S1': 'S1', 
                        'S3': 'S3',
                        'S4': 'S4'
                    }
                    
                    for scenario in scenarios_for_calculation:
                        gcorr_scenario = scenario_mapping[scenario]
                        pd_col = f'gcorr_{gcorr_type}_PD_{gcorr_scenario}'
                        if pd_col in model_data.columns:
                            scenario_values = model_data[pd_col].dropna()
                            if not scenario_values.empty:
                                # Use country-specific peak (max value) for scenario
                                country_peak = scenario_values.max()
                                model_scenario_data[scenario].append(country_peak)
        
        # Calculate scenario statistics for this model
        unique_countries = len(set(countries_with_data))
        
        if unique_countries > 0:
            # Calculate baseline average for ratio calculations
            baseline_peaks = model_scenario_data.get('Baseline', [])
            model_baseline_avg = np.mean(baseline_peaks) if baseline_peaks else historical_baseline_pd
            
            # Only process scenarios for output (excluding baseline) 
            for rank, scenario in enumerate(scenarios_for_output, 1):
                country_peaks = model_scenario_data[scenario]
                
                if country_peaks:
                    # Calculate average of country-specific peaks
                    avg_scenario_pd = np.mean(country_peaks)
                    
                    # For stress scenarios (S1, S3, S4), compare to model baseline
                    vs_baseline_ratio = avg_scenario_pd / model_baseline_avg if model_baseline_avg > 0 else 1
                    
                    scenario_info = {
                        'Model_Type': model_name,
                        'Scenario': scenario,
                        'Countries_Analyzed': len(country_peaks),
                        'Avg_PD_bps': round(avg_scenario_pd * 10000, 1),
                        'Stress_Ratio': round(vs_baseline_ratio, 2)
                    }
                    
                    scenario_analysis.append(scenario_info)
                    print(f"      ✅ {scenario}: {avg_scenario_pd*10000:.1f} bps avg across {len(country_peaks)} countries")
        else:
            print(f"      ⚠️ No data available for {model_name}")
    
    if scenario_analysis:
        scenario_df = pd.DataFrame(scenario_analysis)
        
        # Save to CSV in Final_Report directory
        output_path = Path('Final_Report/table_4_2_scenario_projections.csv')
        output_path.parent.mkdir(exist_ok=True)
        scenario_df.to_csv(output_path, index=False)
        
        print(f"  ✅ Saved scenario projections table: {output_path}")
        print(f"  📊 Models analyzed: {len(scenario_df['Model_Type'].unique())}")
        print(f"  📊 Total scenario projections: {len(scenario_df)}")
        
        return scenario_df
    else:
        print("  ⚠️ No scenario projection data available")
        return None

def generate_regional_scenario_projection_table(all_results, historical_data):
    """Generate regional version of scenario projection table with stress ratios by region"""
    print("\n🌍 Generating Regional Scenario Projection Table (Table 4.2 Regional)...")
    
    regional_scenario_analysis = []
    regional_classifications = get_regional_classifications()
    
    # Define all model types we're analyzing
    model_types = {
        'OVS Historical': 'ovs_historical',
        'OVS GCorr Unsmoothed': 'ovs_gcorr_unsmoothed', 
        'OVS GCorr Smoothed': 'ovs_gcorr_smoothed',
        'GCorr Unsmoothed': 'gcorr_unsmoothed',
        'GCorr Smoothed': 'gcorr_smoothed'
    }
    
    # Define scenarios to analyze (baseline needed for calculations but not included in final table)
    scenarios_for_calculation = ['Baseline', 'S1', 'S3', 'S4']
    scenarios_for_output = ['S1', 'S3', 'S4']  # Exclude baseline from final table
    
    # Calculate historical baseline for each region and add historical crisis analysis
    regional_historical_baselines = {}
    if historical_data is not None:
        # Filter to exclude Serbia only
        historical_data = historical_data[historical_data['cinc'] != 'SRB'].copy()
        
        # Define crisis periods to exclude from baseline calculation
        crisis_periods = [
            (pd.Timestamp('2008-01-01'), pd.Timestamp('2009-12-31')),  # GFC
            (pd.Timestamp('2010-01-01'), pd.Timestamp('2012-12-31')),  # European Crisis
            (pd.Timestamp('2020-01-01'), pd.Timestamp('2021-12-31'))   # COVID
        ]
        
        # Create mask for all crisis periods
        crisis_mask = pd.Series(False, index=historical_data.index)
        for start_date, end_date in crisis_periods:
            period_mask = (
                (historical_data['yyyyqq'] >= start_date) & 
                (historical_data['yyyyqq'] <= end_date)
            )
            crisis_mask = crisis_mask | period_mask
        
        # Calculate non-crisis baseline for each region
        for region, countries in regional_classifications.items():
            region_countries = [c for c in countries if c in historical_data['cinc'].unique()]
            if region_countries:
                region_data = historical_data[
                    historical_data['cinc'].isin(region_countries) & 
                    ~crisis_mask & 
                    historical_data['cdsiedf5'].notna()
                ]
                regional_historical_baselines[region] = region_data['cdsiedf5'].mean() if not region_data.empty else 0.04
                print(f"  📊 {region} historical baseline: {regional_historical_baselines[region]*10000:.1f} bps")
            else:
                regional_historical_baselines[region] = 0.04
        
        # Add regional historical crisis analysis to the scenario table
        print("  📊 Adding regional historical crisis periods to analysis...")
        crisis_info = {
            'Global Financial Crisis': ('2008-2009', pd.Timestamp('2008-01-01'), pd.Timestamp('2009-12-31')),
            'European Debt Crisis': ('2010-2012', pd.Timestamp('2010-01-01'), pd.Timestamp('2012-12-31')),
            'COVID-19 Pandemic': ('2020-2021', pd.Timestamp('2020-01-01'), pd.Timestamp('2021-12-31'))
        }
        
        for crisis_name, (period_desc, start_date, end_date) in crisis_info.items():
            # Calculate crisis impact for each region
            for region, countries in regional_classifications.items():
                region_countries = [c for c in countries if c in historical_data['cinc'].unique()]
                if region_countries:
                    crisis_data = historical_data[
                        (historical_data['yyyyqq'] >= start_date) & 
                        (historical_data['yyyyqq'] <= end_date) &
                        historical_data['cinc'].isin(region_countries) &
                        historical_data['cdsiedf5'].notna()
                    ].copy()
                    
                    if not crisis_data.empty:
                        # Calculate country-specific peaks for this region
                        country_peaks = []
                        for country in region_countries:
                            if country == 'SRB':  # Exclude Serbia only
                                continue
                            country_crisis_data = crisis_data[crisis_data['cinc'] == country]
                            if len(country_crisis_data) > 0:
                                country_peak = country_crisis_data['cdsiedf5'].max()
                                country_peaks.append(country_peak)
                        
                        if country_peaks:
                            # Calculate average of country-specific peaks for this region
                            avg_crisis_peak = np.mean(country_peaks)
                            regional_baseline = regional_historical_baselines.get(region, 0.04)
                            peak_to_baseline_ratio = avg_crisis_peak / regional_baseline if regional_baseline > 0 else 1
                            
                            regional_scenario_analysis.append({
                                'Region': region,
                                'Model_Type': f'Historical Crisis: {crisis_name}',
                                'Scenario': period_desc,
                                'Countries_Analyzed': len(country_peaks),
                                'Avg_PD_bps': round(avg_crisis_peak * 10000, 1),
                                'Stress_Ratio': round(peak_to_baseline_ratio, 2)
                            })
                            
                            print(f"    ✅ {region} {crisis_name}: {avg_crisis_peak*10000:.1f} bps avg peak across {len(country_peaks)} countries (ratio: {peak_to_baseline_ratio:.2f})")
    
    # Process each model type and region combination
    for model_name, model_key in model_types.items():
        print(f"    📊 Processing {model_name} by region...")
        
        # Process each region
        for region, region_countries in regional_classifications.items():
            # Collect data for this model and region across scenarios
            region_scenario_data = {scenario: [] for scenario in scenarios_for_calculation}
            countries_with_data = []
            
            for country in region_countries:
                if country == 'SRB':  # Exclude Serbia only
                    continue
                    
                if country not in all_results:
                    continue
                    
                result = all_results[country]
                
                # Determine data source based on model type
                if model_key.startswith('ovs_'):
                    # OVS model data
                    if not result['has_ovs']:
                        continue
                        
                    # Load OVS forecast data for this country
                    ovs_forecasts = load_ovs_forecasts(country)
                    model_data = ovs_forecasts.get(model_key)
                    
                    if model_data is not None:
                        countries_with_data.append(country)
                        
                        for scenario in scenarios_for_calculation:
                            pd_col = f'{model_key}_PD_{scenario}'
                            if pd_col in model_data.columns:
                                scenario_values = model_data[pd_col].dropna()
                                if not scenario_values.empty:
                                    # Use country-specific peak (max value) for scenario
                                    country_peak = scenario_values.max()
                                    region_scenario_data[scenario].append(country_peak)
                                
                elif model_key.startswith('gcorr_'):
                    # GCorr model data
                    if not result['has_gcorr']:
                        continue
                        
                    gcorr_type = model_key.replace('gcorr_', '')  # 'unsmoothed' or 'smoothed'
                    gcorr_forecasts = load_gcorr_forecasts(country)
                    model_data = gcorr_forecasts.get(gcorr_type)
                    
                    if model_data is not None:
                        countries_with_data.append(country)
                        
                        scenario_mapping = {
                            'Baseline': 'baseline',
                            'S1': 'S1', 
                            'S3': 'S3',
                            'S4': 'S4'
                        }
                        
                        for scenario in scenarios_for_calculation:
                            gcorr_scenario = scenario_mapping[scenario]
                            pd_col = f'gcorr_{gcorr_type}_PD_{gcorr_scenario}'
                            if pd_col in model_data.columns:
                                scenario_values = model_data[pd_col].dropna()
                                if not scenario_values.empty:
                                    # Use country-specific peak (max value) for scenario
                                    country_peak = scenario_values.max()
                                    region_scenario_data[scenario].append(country_peak)
            
            # Calculate scenario statistics for this model-region combination
            unique_countries = len(set(countries_with_data))
            
            if unique_countries > 0:
                # Calculate baseline average for ratio calculations
                baseline_peaks = region_scenario_data.get('Baseline', [])
                region_baseline_avg = np.mean(baseline_peaks) if baseline_peaks else regional_historical_baselines.get(region, 0.04)
                
                # Only process scenarios for output (excluding baseline) 
                for scenario in scenarios_for_output:
                    country_peaks = region_scenario_data[scenario]
                    
                    if country_peaks:
                        # Calculate average of country-specific peaks for this region
                        avg_scenario_pd = np.mean(country_peaks)
                        
                        # For stress scenarios (S1, S3, S4), compare to model baseline
                        vs_baseline_ratio = avg_scenario_pd / region_baseline_avg if region_baseline_avg > 0 else 1
                        
                        scenario_info = {
                            'Region': region,
                            'Model_Type': model_name,
                            'Scenario': scenario,
                            'Countries_Analyzed': len(country_peaks),
                            'Avg_PD_bps': round(avg_scenario_pd * 10000, 1),
                            'Stress_Ratio': round(vs_baseline_ratio, 2)
                        }
                        
                        regional_scenario_analysis.append(scenario_info)
                        print(f"      ✅ {region} {scenario}: {avg_scenario_pd*10000:.1f} bps avg across {len(country_peaks)} countries")
    
    if regional_scenario_analysis:
        regional_scenario_df = pd.DataFrame(regional_scenario_analysis)
        
        # Sort by Region, then Model_Type, then Scenario
        regional_scenario_df = regional_scenario_df.sort_values(['Region', 'Model_Type', 'Scenario'])
        
        # Save to CSV in Final_Report directory
        output_path = Path('Final_Report/table_4_2_regional_scenario_projections.csv')
        output_path.parent.mkdir(exist_ok=True)
        regional_scenario_df.to_csv(output_path, index=False)
        
        print(f"  ✅ Saved regional scenario projections table: {output_path}")
        print(f"  🌍 Regions analyzed: {len(regional_scenario_df['Region'].unique())}")
        print(f"  📊 Models analyzed: {len(regional_scenario_df['Model_Type'].unique())}")
        print(f"  📊 Total regional scenario projections: {len(regional_scenario_df)}")
        
        return regional_scenario_df
    else:
        print("  ⚠️ No regional scenario projection data available")
        return None

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
    
    # Generate scenario projection table (Table 4.2) using the loaded data
    scenario_table = generate_scenario_projection_table(all_results, historical_data)
    
    # Generate regional scenario projection table (Table 4.2 Regional)
    regional_scenario_table = generate_regional_scenario_projection_table(all_results, historical_data)
    
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