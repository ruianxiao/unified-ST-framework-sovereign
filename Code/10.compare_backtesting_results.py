import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
HORIZON_QUARTERS = 4  # Focus on 4Q backtesting
GCORR_FORECAST_QUARTERS = 40  # Number of GCorr forecast quarters (20 or 40)

# Data paths
OVS_BACKTESTING_DIR = Path('Output/8.backtesting_analysis')
GCORR_BACKTESTING_DIR = Path('gcorr-sovereign-backtesting/Output')

# Output directories
OUTPUT_DIR = Path('Output/10.backtesting_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations to compare (based on available backtesting results)
OVS_MODELS = {
    'OVS_Historical': 'advanced',
    'OVS_GCorr_Unsmoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q',
    'OVS_GCorr_Smoothed': f'advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed'
}

GCORR_MODELS = {
    'GCorr_Unsmoothed': 'GCorr Stressed PD (Unsmoothed)',
    'GCorr_Smoothed': 'GCorr Stressed PD (Smoothed)'
}

# Model colors for consistent plotting
MODEL_COLORS = {
    'Historical': 'black',
    'OVS_Historical': 'blue',
    'OVS_GCorr_Unsmoothed': 'green',
    'OVS_GCorr_Smoothed': 'orange',
    'GCorr_Unsmoothed': 'red',
    'GCorr_Smoothed': 'purple'
}

def load_ovs_backtesting_results(country):
    """Load OVS backtesting results for a specific country"""
    ovs_results = {}
    
    country_dir = OVS_BACKTESTING_DIR / country
    if not country_dir.exists():
        return ovs_results
    
    for model_name, model_source in OVS_MODELS.items():
        backtest_file = country_dir / f'backtest_results_{model_source}_{HORIZON_QUARTERS}Q.csv'
        
        if not backtest_file.exists():
            print(f"  OVS backtest file not found: {backtest_file.name}")
            continue
        
        try:
            # Load backtesting results
            backtest_data = pd.read_csv(backtest_file)
            backtest_data['yyyyqq'] = pd.to_datetime(backtest_data['yyyyqq'])
            backtest_data['forecast_start_date'] = pd.to_datetime(backtest_data['forecast_start_date'])
            
            # Calculate cumulative backtesting results similar to the plot function
            forecast_starts = sorted(backtest_data['forecast_start_date'].unique())
            cumulative_results = []
            
            for start_date in forecast_starts:
                start_data = backtest_data[backtest_data['forecast_start_date'] == start_date].copy()
                
                if start_data.empty:
                    continue
                
                # Sort by quarter ahead to ensure proper cumulation
                if 'quarter_ahead' in start_data.columns:
                    start_data = start_data.sort_values('quarter_ahead')
                
                # Convert annualized PD to quarterly PD, then sum across all quarters in the horizon
                # Using the more accurate compound formula: quarterly_PD = 1 - (1 - annualized_PD)^(1/4)
                # Handle missing values properly - they should remain as NaN
                quarterly_predicted = 1 - (1 - start_data['predicted_PD'])**(1/4)
                quarterly_actual = 1 - (1 - start_data['actual_PD'])**(1/4)
                
                # Calculate cumulative sum of quarterly PD across all quarters in the horizon
                # If any quarter has NaN, the cumulative result should be NaN
                cumulative_predicted = quarterly_predicted.sum() if not pd.isna(quarterly_predicted).any() else np.nan
                cumulative_actual = quarterly_actual.sum() if not pd.isna(quarterly_actual).any() else np.nan
                
                cumulative_results.append({
                    'forecast_start_date': start_date,
                    'cumulative_predicted_PD': cumulative_predicted,
                    'cumulative_actual_PD': cumulative_actual,
                    'max_horizon': start_data['quarter_ahead'].max(),
                    'n_quarters': len(start_data)
                })
            
            if cumulative_results:
                cum_df = pd.DataFrame(cumulative_results)
                cum_df = cum_df.sort_values('forecast_start_date')
                ovs_results[model_name] = cum_df
            
        except Exception as e:
            print(f"  Error loading OVS backtest for {country} ({model_name}): {e}")
            continue
    
    return ovs_results

def load_aligned_backtesting_results(country):
    """Load and align all backtesting results to GCorr quarters timeline"""
    aligned_results = {}
    
    # First load GCorr data to get the reference timeline
    gcorr_file = GCORR_BACKTESTING_DIR / f'{country}_backtesting_results.csv'
    if not gcorr_file.exists():
        print(f"  GCorr backtest file not found: {gcorr_file.name}")
        return aligned_results
    
    try:
        # Load GCorr backtesting results
        gcorr_data = pd.read_csv(gcorr_file)
        
        # Convert GCorr date format "2006 Q1" to datetime (last day of quarter)
        def convert_gcorr_date(date_str):
            year, quarter = date_str.split(' Q')
            quarter_end = {'1': '03-31', '2': '06-30', '3': '09-30', '4': '12-31'}
            return pd.to_datetime(f"{year}-{quarter_end[quarter]}")
        
        gcorr_data['yyyyqq'] = gcorr_data['yyyyqq'].apply(convert_gcorr_date)
        gcorr_data = gcorr_data.sort_values('yyyyqq')
        
        # Create aligned results using GCorr quarters as reference
        forecast_quarters = []
        
        for i in range(1, len(gcorr_data) - HORIZON_QUARTERS):
            forecast_start = gcorr_data.iloc[i]['yyyyqq']
            current_data = gcorr_data.iloc[i]
            
            # GCorr data (already cumulative)
            gcorr_actual = current_data["Moody's Analytics CDS-implied EDF"]
            gcorr_unsmoothed = current_data['GCorr Stressed PD (Unsmoothed)']
            gcorr_smoothed = current_data['GCorr Stressed PD (Smoothed)']
            
            # Initialize OVS data as NaN
            ovs_historical = np.nan
            ovs_gcorr_unsmoothed = np.nan
            ovs_gcorr_smoothed = np.nan
            
            forecast_quarters.append({
                'forecast_start_date': forecast_start,
                'gcorr_actual_PD': gcorr_actual,
                'gcorr_unsmoothed_PD': gcorr_unsmoothed,
                'gcorr_smoothed_PD': gcorr_smoothed,
                'ovs_historical_PD': ovs_historical,
                'ovs_gcorr_unsmoothed_PD': ovs_gcorr_unsmoothed,
                'ovs_gcorr_smoothed_PD': ovs_gcorr_smoothed
            })
        
        if not forecast_quarters:
            return aligned_results
            
        aligned_df = pd.DataFrame(forecast_quarters)
        
        # Now try to fill in OVS data for matching quarters
        ovs_data = load_ovs_backtesting_results(country)
        
        for ovs_model, ovs_results in ovs_data.items():
            if ovs_results is not None and not ovs_results.empty:
                # Map OVS model names to column names
                ovs_column_map = {
                    'OVS_Historical': 'ovs_historical_PD',
                    'OVS_GCorr_Unsmoothed': 'ovs_gcorr_unsmoothed_PD', 
                    'OVS_GCorr_Smoothed': 'ovs_gcorr_smoothed_PD'
                }
                
                if ovs_model in ovs_column_map:
                    ovs_col = ovs_column_map[ovs_model]
                    
                    # Merge OVS data with aligned data based on exact forecast start date match
                    for idx, row in aligned_df.iterrows():
                        gcorr_date = row['forecast_start_date']
                        
                        # Find exact match for OVS forecast date
                        exact_matches = ovs_results[ovs_results['forecast_start_date'] == gcorr_date]
                        
                        # Only use if exact match exists
                        if not exact_matches.empty:
                            aligned_df.loc[idx, ovs_col] = exact_matches.iloc[0]['cumulative_predicted_PD']
        
        # Create results in the expected format
        # GCorr models
        aligned_results['GCorr_Unsmoothed'] = pd.DataFrame({
            'forecast_start_date': aligned_df['forecast_start_date'],
            'cumulative_actual_PD': aligned_df['gcorr_actual_PD'],
            'cumulative_predicted_PD': aligned_df['gcorr_unsmoothed_PD'],
            'max_horizon': HORIZON_QUARTERS,
            'n_quarters': HORIZON_QUARTERS
        })
        
        aligned_results['GCorr_Smoothed'] = pd.DataFrame({
            'forecast_start_date': aligned_df['forecast_start_date'],
            'cumulative_actual_PD': aligned_df['gcorr_actual_PD'],
            'cumulative_predicted_PD': aligned_df['gcorr_smoothed_PD'],
            'max_horizon': HORIZON_QUARTERS,
            'n_quarters': HORIZON_QUARTERS
        })
        
        # OVS models (with GCorr actual PD for consistency)
        for ovs_model, ovs_col in [('OVS_Historical', 'ovs_historical_PD'),
                                   ('OVS_GCorr_Unsmoothed', 'ovs_gcorr_unsmoothed_PD'),
                                   ('OVS_GCorr_Smoothed', 'ovs_gcorr_smoothed_PD')]:
            if not all(pd.isna(aligned_df[ovs_col])):  # Only include if we have some data
                aligned_results[ovs_model] = pd.DataFrame({
                    'forecast_start_date': aligned_df['forecast_start_date'],
                    'cumulative_actual_PD': aligned_df['gcorr_actual_PD'],  # Use GCorr actual for consistency
                    'cumulative_predicted_PD': aligned_df[ovs_col],
                    'max_horizon': HORIZON_QUARTERS,
                    'n_quarters': HORIZON_QUARTERS
                })
    
    except Exception as e:
        print(f"  Error loading aligned backtest for {country}: {e}")
        return aligned_results
    
    return aligned_results

def get_model_metadata(country):
    """Get metadata about models for a country"""
    metadata = {'country': country}
    
    # Get OVS metadata from master summary
    try:
        master_summary = pd.read_csv(OVS_BACKTESTING_DIR / 'master_backtesting_summary_all_combinations.csv')
        
        for model_name, model_source in OVS_MODELS.items():
            combination_name = f"{model_source}_{HORIZON_QUARTERS}Q"
            country_data = master_summary[
                (master_summary['country'] == country) & 
                (master_summary['combination'] == combination_name)
            ]
            
            if not country_data.empty:
                row = country_data.iloc[0]
                metadata[f'{model_name}_adj_r2'] = row['model_adj_r2']
                metadata[f'{model_name}_correlation'] = row['pd_correlation']
                metadata[f'{model_name}_mape'] = row['pd_mape']
            else:
                # Handle cases where the combination might not be in master summary
                # but the individual files exist (common for 'advanced' historical)
                if model_name == 'OVS_Historical':
                    print(f"  Note: {model_name} not in master summary for {country}, but file may exist")
                # Set default values for any missing models
                metadata[f'{model_name}_adj_r2'] = 'N/A'
                metadata[f'{model_name}_correlation'] = 'N/A'
                metadata[f'{model_name}_mape'] = 'N/A'
    
    except Exception as e:
        print(f"  Warning: Could not load OVS metadata for {country}: {e}")
    
    # Get GCorr metadata from summary statistics
    try:
        gcorr_summary = pd.read_csv(GCORR_BACKTESTING_DIR / 'backtesting_summary_statistics.csv')
        country_data = gcorr_summary[gcorr_summary['country'] == country]
        
        if not country_data.empty:
            row = country_data.iloc[0]
            metadata['GCorr_Unsmoothed_correlation'] = row['correlation_unsmoothed']
            metadata['GCorr_Smoothed_correlation'] = row['correlation_smoothed']
            metadata['GCorr_Unsmoothed_rmse'] = row['rmse_unsmoothed']
            metadata['GCorr_Smoothed_rmse'] = row['rmse_smoothed']
    
    except Exception as e:
        print(f"  Warning: Could not load GCorr metadata for {country}: {e}")
    
    return metadata

def create_backtesting_comparison_plot(country, ovs_results, gcorr_results, metadata):
    """Create comparison plot for backtesting results"""
    
    # Set much larger font sizes for all plot elements (Word document readability)
    plt.rcParams.update({
        'font.size': 18,           # Base font size
        'axes.titlesize': 20,      # Subplot titles
        'axes.labelsize': 22,      # Axis labels (increased)
        'xtick.labelsize': 18,     # X-axis tick labels (increased)
        'ytick.labelsize': 18,     # Y-axis tick labels (increased)
        'legend.fontsize': 22,     # Legend text (increased)
        'figure.titlesize': 24     # Main title
    })
    
    # Create figure with larger size for better readability
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    # Remove main title - information will be in subplot titles
    
    # Collect all data for consistent y-axis scaling
    all_pd_values = []
    all_models_data = {}
    
    # Process OVS results
    for model_name, model_data in ovs_results.items():
        if model_data is not None and not model_data.empty:
            all_pd_values.extend(model_data['cumulative_predicted_PD'].dropna().tolist())
            all_pd_values.extend(model_data['cumulative_actual_PD'].dropna().tolist())
            all_models_data[model_name] = model_data
    
    # Process GCorr results
    for model_name, model_data in gcorr_results.items():
        if model_data is not None and not model_data.empty:
            all_pd_values.extend(model_data['cumulative_predicted_PD'].dropna().tolist())
            all_pd_values.extend(model_data['cumulative_actual_PD'].dropna().tolist())
            all_models_data[model_name] = model_data
    
    # Calculate consistent y-axis limits
    if all_pd_values:
        y_min, y_max = min(all_pd_values), max(all_pd_values)
        
        # Add 10% padding
        y_range = y_max - y_min
        y_padding = max(y_range * 0.1, y_max * 0.05)
        y_min_plot = max(0, y_min - y_padding)
        y_max_plot = y_max + y_padding
    else:
        y_min_plot, y_max_plot = 0, 0.01
    
    # Plot 1: Smoothed Models Comparison (OVS Historical vs OVS GCorr Smoothed vs GCorr Smoothed)
    ax1 = axes[0]
    smoothed_title = f'Backtesting Results: {country} ({HORIZON_QUARTERS}Q) - Smoothed Models'
    smoothed_stats = plot_methodology_comparison(ax1, smoothed_title, all_models_data, 
                                               ['OVS_Historical', 'OVS_GCorr_Smoothed', 'GCorr_Smoothed'],
                                               y_min_plot, y_max_plot, metadata)
    
    # Plot 2: Unsmoothed Models Comparison (OVS Historical vs OVS GCorr Unsmoothed vs GCorr Unsmoothed)
    ax2 = axes[1]
    unsmoothed_title = f'Backtesting Results: {country} ({HORIZON_QUARTERS}Q) - Unsmoothed Models'
    unsmoothed_stats = plot_methodology_comparison(ax2, unsmoothed_title, all_models_data, 
                                                 ['OVS_Historical', 'OVS_GCorr_Unsmoothed', 'GCorr_Unsmoothed'],
                                                 y_min_plot, y_max_plot, metadata)
    
    # Combine statistics from both plots
    country_stats = {**smoothed_stats, **unsmoothed_stats}
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.15)  # More space for charts since no main title
    
    # Save plot
    plt.savefig(PLOTS_DIR / f'backtesting_comparison_{country}_{HORIZON_QUARTERS}Q.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib rcParams to defaults
    plt.rcdefaults()
    
    print(f"  Saved backtesting comparison plot for {country}")
    
    return country_stats

def plot_methodology_comparison(ax, title, all_models_data, model_names, y_min_plot, y_max_plot, metadata):
    """Plot comparison of multiple methodologies on the same chart"""
    
    # Plot actual data once (use the first available model's actual data)
    actual_data = None
    for model_name in model_names:
        if model_name in all_models_data:
            actual_data = all_models_data[model_name]
            break
    
    if actual_data is None:
        ax.text(0.5, 0.5, f'No data available for {title}', 
                transform=ax.transAxes, ha='center', va='center', fontsize=18)
        ax.set_title(title, fontsize=20, fontweight='bold')
        return {}
    
    # Plot actual PD (handle missing values to show gaps) - with thicker lines
    actual_pd_values = actual_data['cumulative_actual_PD'] * 100
    actual_pd_values = actual_pd_values.where(actual_pd_values.notna(), None)
    ax.plot(actual_data['forecast_start_date'], actual_pd_values, 
            color='black', linestyle='--', linewidth=3, alpha=0.8,
            label='Actual', marker='s', markersize=8)
    
    # Plot each model's predictions and calculate statistics
    model_stats = {}
    stat_labels = []
    
    for model_name in model_names:
        if model_name in all_models_data:
            model_data = all_models_data[model_name]
            model_label = model_name.replace('_', ' ')
            
            # Use different colors for each model
            color = MODEL_COLORS.get(model_name, 'blue')
            
            # Handle missing values to show gaps in predicted PD - with thicker lines
            predicted_pd_values = model_data['cumulative_predicted_PD'] * 100
            predicted_pd_values = predicted_pd_values.where(predicted_pd_values.notna(), None)
            ax.plot(model_data['forecast_start_date'], predicted_pd_values, 
                    color=color, linestyle='-', linewidth=4, alpha=0.9,
                    label=model_label, marker='o', markersize=8)
            
            # Calculate statistics for non-missing data points
            mask = model_data['cumulative_predicted_PD'].notna() & model_data['cumulative_actual_PD'].notna()
            if mask.sum() > 0:
                predicted = model_data.loc[mask, 'cumulative_predicted_PD']
                actual = model_data.loc[mask, 'cumulative_actual_PD']
                
                # Calculate correlation
                corr = predicted.corr(actual)
                
                # Calculate MAPE
                mape = np.mean(np.abs((predicted - actual) / actual)) * 100
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((predicted - actual) ** 2))
                
                model_stats[model_name] = {
                    'correlation': corr,
                    'mape': mape, 
                    'rmse': rmse
                }
                
                stat_labels.append(f"{model_label}: Corr={corr:.3f}, MAPE={mape:.1f}%, RMSE={rmse:.4f}")
    
    # Set title with statistics (one line per model) - use larger fonts
    subtitle = '\n'.join(stat_labels)
    ax.set_title(f'{title}\n{subtitle}', fontsize=22)
    # Remove x-axis title as requested
    ax.set_ylabel('Cumulative PD (%)', fontsize=22, fontweight='bold')
    ax.legend(fontsize=22, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_ylim(y_min_plot * 100, y_max_plot * 100)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    
    return model_stats


def plot_gcorr_comparison(ax, title, all_models_data, y_min_plot, y_max_plot, metadata):
    """Plot GCorr models comparison"""
    gcorr_models = ['GCorr_Unsmoothed', 'GCorr_Smoothed']
    
    # Plot actual data once
    actual_data = None
    for model_name in gcorr_models:
        if model_name in all_models_data:
            actual_data = all_models_data[model_name]
            break
    
    if actual_data is None:
        ax.text(0.5, 0.5, 'No GCorr data available', 
                transform=ax.transAxes, ha='center', va='center', fontsize=18)
        ax.set_title(title, fontsize=20, fontweight='bold')
        return {}
    
    # Plot actual PD (handle missing values to show gaps) - with thicker lines
    actual_pd_values = actual_data['cumulative_actual_PD'] * 100
    actual_pd_values = actual_pd_values.where(actual_pd_values.notna(), None)
    ax.plot(actual_data['forecast_start_date'], actual_pd_values, 
            color='black', linestyle='--', linewidth=3, alpha=0.8,
            label='Actual', marker='s', markersize=8)
    
    # Plot GCorr predictions and calculate statistics
    model_stats = {}
    stat_labels = []
    
    for model_name in gcorr_models:
        if model_name in all_models_data:
            model_data = all_models_data[model_name]
            model_label = model_name.replace('_', ' ')
            
            # Handle missing values to show gaps in predicted PD - with thicker lines
            predicted_pd_values = model_data['cumulative_predicted_PD'] * 100
            predicted_pd_values = predicted_pd_values.where(predicted_pd_values.notna(), None)
            ax.plot(model_data['forecast_start_date'], predicted_pd_values, 
                    color=MODEL_COLORS[model_name], linestyle='-', linewidth=4, alpha=0.9,
                    label=model_label, marker='o', markersize=8)
            
            # Calculate statistics for non-missing data points
            mask = model_data['cumulative_predicted_PD'].notna() & model_data['cumulative_actual_PD'].notna()
            if mask.sum() > 0:
                predicted = model_data.loc[mask, 'cumulative_predicted_PD']
                actual = model_data.loc[mask, 'cumulative_actual_PD']
                
                # Calculate correlation
                corr = predicted.corr(actual)
                
                # Calculate MAPE
                mape = np.mean(np.abs((predicted - actual) / actual)) * 100
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((predicted - actual) ** 2))
                
                model_stats[model_name] = {
                    'correlation': corr,
                    'mape': mape, 
                    'rmse': rmse
                }
                
                stat_labels.append(f"{model_label}: Corr={corr:.3f}, MAPE={mape:.1f}%, RMSE={rmse:.4f}")
    
    # Set title with statistics (one line per model) - use larger fonts
    subtitle = '\n'.join(stat_labels)
    ax.set_title(f'{title}\n{subtitle}', fontsize=22)
    # Remove x-axis title as requested
    ax.set_ylabel('Cumulative PD (%)', fontsize=22, fontweight='bold')
    ax.legend(fontsize=22, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_ylim(y_min_plot * 100, y_max_plot * 100)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    
    return model_stats



def calculate_comparison_metrics(all_results):
    """Calculate comprehensive comparison metrics across all countries"""
    metrics_data = []
    
    for country, result in all_results.items():
        if not result['has_data']:
            continue
            
        record = {
            'country': country,
            # Model availability flags
            'has_ovs_historical': 'OVS_Historical' in result.get('aligned_results', {}),
            'has_ovs_gcorr_unsmoothed': 'OVS_GCorr_Unsmoothed' in result.get('aligned_results', {}),
            'has_ovs_gcorr_smoothed': 'OVS_GCorr_Smoothed' in result.get('aligned_results', {}),
            'has_gcorr_unsmoothed': 'GCorr_Unsmoothed' in result.get('aligned_results', {}),
            'has_gcorr_smoothed': 'GCorr_Smoothed' in result.get('aligned_results', {})
        }
        
        # Add detailed statistics for each model (grouped by model)
        statistics = result.get('statistics', {})
        
        # OVS Historical statistics
        if 'OVS_Historical' in statistics:
            stats = statistics['OVS_Historical']
            record['ovs_historical_correlation'] = stats['correlation']
            record['ovs_historical_mape'] = stats['mape']
            record['ovs_historical_rmse'] = stats['rmse']
        else:
            record['ovs_historical_correlation'] = None
            record['ovs_historical_mape'] = None
            record['ovs_historical_rmse'] = None
        
        # OVS GCorr Unsmoothed statistics
        if 'OVS_GCorr_Unsmoothed' in statistics:
            stats = statistics['OVS_GCorr_Unsmoothed']
            record['ovs_gcorr_unsmoothed_correlation'] = stats['correlation']
            record['ovs_gcorr_unsmoothed_mape'] = stats['mape']
            record['ovs_gcorr_unsmoothed_rmse'] = stats['rmse']
        else:
            record['ovs_gcorr_unsmoothed_correlation'] = None
            record['ovs_gcorr_unsmoothed_mape'] = None
            record['ovs_gcorr_unsmoothed_rmse'] = None
        
        # OVS GCorr Smoothed statistics
        if 'OVS_GCorr_Smoothed' in statistics:
            stats = statistics['OVS_GCorr_Smoothed']
            record['ovs_gcorr_smoothed_correlation'] = stats['correlation']
            record['ovs_gcorr_smoothed_mape'] = stats['mape']
            record['ovs_gcorr_smoothed_rmse'] = stats['rmse']
        else:
            record['ovs_gcorr_smoothed_correlation'] = None
            record['ovs_gcorr_smoothed_mape'] = None
            record['ovs_gcorr_smoothed_rmse'] = None
        
        # GCorr Unsmoothed statistics
        if 'GCorr_Unsmoothed' in statistics:
            stats = statistics['GCorr_Unsmoothed']
            record['gcorr_unsmoothed_correlation'] = stats['correlation']
            record['gcorr_unsmoothed_mape'] = stats['mape']
            record['gcorr_unsmoothed_rmse'] = stats['rmse']
        else:
            record['gcorr_unsmoothed_correlation'] = None
            record['gcorr_unsmoothed_mape'] = None
            record['gcorr_unsmoothed_rmse'] = None
        
        # GCorr Smoothed statistics
        if 'GCorr_Smoothed' in statistics:
            stats = statistics['GCorr_Smoothed']
            record['gcorr_smoothed_correlation'] = stats['correlation']
            record['gcorr_smoothed_mape'] = stats['mape']
            record['gcorr_smoothed_rmse'] = stats['rmse']
        else:
            record['gcorr_smoothed_correlation'] = None
            record['gcorr_smoothed_mape'] = None
            record['gcorr_smoothed_rmse'] = None
        
        metrics_data.append(record)
    
    return pd.DataFrame(metrics_data)

def main():
    print("Backtesting Results Comparison")
    print(f"Horizon: {HORIZON_QUARTERS} quarters")
    print(f"OVS Models: {list(OVS_MODELS.keys())}")
    print(f"GCorr Models: {list(GCORR_MODELS.keys())}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Check if required directories exist
    if not OVS_BACKTESTING_DIR.exists():
        print(f"ERROR: OVS backtesting directory not found: {OVS_BACKTESTING_DIR}")
        return
    if not GCORR_BACKTESTING_DIR.exists():
        print(f"ERROR: GCorr backtesting directory not found: {GCORR_BACKTESTING_DIR}")
        return
    
    # Get list of countries from both sources
    ovs_countries = set()
    gcorr_countries = set()
    
    # Get OVS countries
    for country_dir in OVS_BACKTESTING_DIR.iterdir():
        if country_dir.is_dir() and country_dir.name not in ['plots', 'comparison_charts_backtesting']:
            ovs_countries.add(country_dir.name)
    
    # Get GCorr countries
    for file in GCORR_BACKTESTING_DIR.glob('*_backtesting_results.csv'):
        country = file.stem.replace('_backtesting_results', '')
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
    
    for i, country in enumerate(sorted(all_countries), 1):
        print(f"\n[{i}/{len(all_countries)}] Processing {country}...")
        
        try:
            # Load aligned data (includes both OVS and GCorr results aligned to same timeline)
            aligned_results = load_aligned_backtesting_results(country)
            metadata = get_model_metadata(country)
            
            # Check if we have any data
            has_data = bool(aligned_results)
            
            if has_data:
                # Split aligned results into OVS and GCorr for plotting
                ovs_results = {k: v for k, v in aligned_results.items() if k.startswith('OVS_')}
                gcorr_results = {k: v for k, v in aligned_results.items() if k.startswith('GCorr_')}
                
                # Create comparison plot and collect statistics
                country_stats = create_backtesting_comparison_plot(country, ovs_results, gcorr_results, metadata)
                
                successful_comparisons += 1
            else:
                print(f"  No backtesting data available for {country}")
            
            all_results[country] = {
                'has_data': has_data,
                'aligned_results': aligned_results,
                'metadata': metadata,
                'statistics': country_stats if has_data else {}
            }
            
        except Exception as e:
            print(f"  Error processing {country}: {e}")
            continue
    
    # Calculate and save comprehensive comparison metrics
    metrics_df = calculate_comparison_metrics(all_results)
    metrics_df.to_csv(OUTPUT_DIR / 'backtesting_comparison_summary.csv', index=False)
    print(f"Saved comprehensive comparison summary: {len(metrics_df)} countries with detailed statistics")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BACKTESTING COMPARISON COMPLETE!")
    print(f"Successfully created comparison plots: {successful_comparisons}")
    print(f"Countries with both OVS and GCorr data: {len(common_countries)}")
    print(f"Countries with only OVS data: {len(ovs_countries - gcorr_countries)}")
    print(f"Countries with only GCorr data: {len(gcorr_countries - ovs_countries)}")
    
    if not metrics_df.empty:
        print(f"\nModel Coverage:")
        print(f"OVS Historical: {metrics_df['has_ovs_historical'].sum()} countries")
        print(f"OVS GCorr Unsmoothed: {metrics_df['has_ovs_gcorr_unsmoothed'].sum()} countries")
        print(f"GCorr Unsmoothed: {metrics_df['has_gcorr_unsmoothed'].sum()} countries")
        print(f"GCorr Smoothed: {metrics_df['has_gcorr_smoothed'].sum()} countries")
    
    print(f"\nOutput files:")
    print(f"- Comparison plots: {PLOTS_DIR}")
    print(f"- Comprehensive summary: backtesting_comparison_summary.csv")
    print(f"- Horizon: {HORIZON_QUARTERS} quarters")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 