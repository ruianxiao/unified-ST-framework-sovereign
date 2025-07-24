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
output_dir = Path('Output/8.backtesting_analysis')
output_dir.mkdir(parents=True, exist_ok=True)
plots_dir = output_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

# Create centralized comparison directory for backtesting charts
comparison_plots_dir = output_dir / 'comparison_charts_backtesting'
comparison_plots_dir.mkdir(parents=True, exist_ok=True)

# Configuration
DATA_PATH = 'Output/4.transformation/transformed_data.csv'
OVS_RESULTS_DIR = Path('Output/6.ovs_variable_selection')

# Backtesting configuration
BACKTESTING_HORIZONS = [4, 8, 12]  # N quarters ahead to forecast (1 year, 2 years, 3 years)
MIN_LAG_QUARTERS = 1  # Minimum quarters needed for lag variables (since we use pre-trained coefficients)
DEFAULT_HORIZON = 8  # Default N quarters for backtesting

# Use only advanced filtered results and no lags (simplified from original)
OVS_SOURCES = {
    'advanced': 'Output/7.filtered_ovs_results/filtered_ovs_results_advanced.csv',
    'advanced_gcorr40q': 'Output/7.filtered_ovs_results/filtered_ovs_results_advanced_gcorr40q.csv',
    'advanced_gcorr40q_smoothed': 'Output/7.filtered_ovs_results/filtered_ovs_results_advanced_gcorr40q_smoothed.csv'
}

# Only use no_lags option (simplified from original)
LAG_OPTION = 'has_no_lags'

def get_top_model_for_country(country, ovs_source, lag_option='has_no_lags'):
    """
    Helper function to get the top model for a country from OVS results.
    Returns the model row or None if not found.
    """
    try:
        ovs_file_path = OVS_SOURCES[ovs_source]
        ovs_data = pd.read_csv(ovs_file_path)
        
        if ovs_source in ['original', 'original_gcorr40q', 'original_gcorr40q_smoothed']:
            # For original sources, get the top model by adj_r2
            country_models = ovs_data[ovs_data['country'] == country]
            
            # Apply has_no_lags filter if requested
            if lag_option == 'has_no_lags' and 'has_no_lags' in ovs_data.columns:
                country_models = country_models[country_models['has_no_lags'] == True]
                
            if not country_models.empty:
                return country_models.loc[country_models['adj_r2'].idxmax()]
        else:
            # For filtered results, first filter by country
            country_models = ovs_data[ovs_data['country'] == country]
            
            # Apply has_no_lags filter if requested
            if lag_option == 'has_no_lags' and 'has_no_lags' in ovs_data.columns:
                country_models = country_models[country_models['has_no_lags'] == True]
            
            # Get the best model (lowest rank) from the filtered set
            if not country_models.empty:
                best_model = country_models.loc[country_models['rank_in_country'].idxmin()]
                return best_model
        
        return None
        
    except Exception as e:
        print(f"  Warning: Could not load model for {country} from {ovs_source}: {str(e)}")
        return None

def get_model_description_for_plotting(country, ovs_source):
    """
    Helper function to create model description string for plot titles.
    Returns formatted string with variable names and coefficients.
    """
    model_row = get_top_model_for_country(country, ovs_source)
    
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
    
    # Add macro variables with coefficients (no lags)
    for i in range(1, 5):  # MV1 to MV4
        mv_col = f'MV{i}'
        coeff_col = f'MV{i}_coefficient'
        
        if (mv_col in model_row and pd.notna(model_row[mv_col]) and 
            coeff_col in model_row and pd.notna(model_row[coeff_col])):
            
            mv_name = model_row[mv_col]
            coeff = model_row[coeff_col]
            clean_mv_names.append(f"{mv_name}({coeff:.2f})")
    
    return ' + '.join(clean_mv_names) if clean_mv_names else 'Model variables not available'

def load_ovs_results(ovs_source='advanced', lag_option='has_no_lags'):
    """Load OVS results and return top model for each country (has_no_lags version)"""
    ovs_results = {}
    
    ovs_file = Path(OVS_SOURCES[ovs_source])
    
    if not ovs_file.exists():
        raise FileNotFoundError(f"OVS results file not found: {ovs_file}")
    
    all_results = pd.read_csv(ovs_file)
    
    # Process each country to get the appropriate model based on lag_option
    countries = all_results['country'].unique()
    processed_countries = 0
    
    for country in countries:
        # Get the appropriate model for this country based on lag_option
        model = get_top_model_for_country(country, ovs_source, lag_option)
        if model is None:
            continue
        
        processed_countries += 1
        
        # Build model_vars and coefficients
        model_vars = []
        coefficients = []
        
        # Add intercept
        if 'constant_coefficient' in model.index:
            const_coeff = model.get('constant_coefficient')
            if pd.notna(const_coeff):
                model_vars.append('const')
                coefficients.append(const_coeff)
        
        # Add dlnPD lags
        if 'includes_lag' in model.index and 'lag_coefficient' in model.index:
            includes_lag = model.get('includes_lag')
            lag_coeff = model.get('lag_coefficient')
            if pd.notna(includes_lag) and includes_lag == True and pd.notna(lag_coeff):
                model_vars.append('dlnPD_l1')  # The OVS uses lag1 for dlnPD
                coefficients.append(lag_coeff)
        
        # Add mean reverting term
        if 'mean_reverting_coefficient' in model.index:
            mean_rev_coeff = model.get('mean_reverting_coefficient')
            if pd.notna(mean_rev_coeff):
                model_vars.append('mean_reverting')
                coefficients.append(mean_rev_coeff)
        
        # Add macro variables - always use no lags (Baseline_trans)
        for i in range(1, 5):  # MV1-MV4
            mv_col = f'MV{i}'
            coeff_col = f'MV{i}_coefficient'
            
            if mv_col in model.index and pd.notna(model.get(mv_col)):
                mv_value = model.get(mv_col)
                if isinstance(mv_value, str) and mv_value.strip():
                    base_mv_name = mv_value.split('_lag')[0] if '_lag' in mv_value else mv_value
                    # Always use unshifted macro variable (no lags)
                    mv_name = f"{base_mv_name}_Baseline_trans"
                    model_vars.append(mv_name)
                    coefficients.append(model[coeff_col])
        
        ovs_results[country] = {
            'model_vars': model_vars,
            'coefficients': coefficients,
            'adj_r2': model['adj_r2'],
            'rank': model.get('rank_in_country', 1) if 'rank_in_country' in model.index else 1,
            'has_no_lags': model.get('has_no_lags', False),
            'ovs_source': ovs_source,
            'lag_option': lag_option
        }
    
    if processed_countries == 0:
        print(f"  No suitable models found in {ovs_source} with lag_option={lag_option}. Skipping backtesting for this source.")
        return {}
    
    print(f"Loaded OVS results for {len(ovs_results)} countries (source: {ovs_source})")
    return ovs_results

def forecast_dlnpd_backtesting(data, model_info, start_idx, horizon_quarters):
    """
    Forecast dlnPD for N quarters ahead using realized MV data for backtesting.
    
    Args:
        data: DataFrame with historical data sorted by date
        model_info: Dictionary with model variables and coefficients
        start_idx: Index in data to start forecasting from
        horizon_quarters: Number of quarters to forecast ahead (N)
    
    Returns:
        DataFrame with forecasted and actual PD values
    """
    model_vars = model_info['model_vars']
    coefficients = model_info['coefficients']
    
    # Convert coefficients list to dictionary for easier access
    coeff_dict = dict(zip(model_vars, coefficients))
    
    # Ensure we have enough data for the horizon
    if start_idx + horizon_quarters >= len(data):
        return pd.DataFrame()
    
    # Initialize forecast results
    forecast_results = []
    
    # Get the starting point PD for initialization
    start_pd = data.iloc[start_idx]['cdsiedf5'] if pd.notna(data.iloc[start_idx]['cdsiedf5']) else None
    if start_pd is None or start_pd <= 0:
        return pd.DataFrame()
    
    current_lnpd = np.log(start_pd)
    current_dlnpd = 0
    
    # Generate recursive forecasts for N quarters
    for q in range(1, horizon_quarters + 1):
        target_idx = start_idx + q
        
        # Update lagged dlnPD variables with recent predictions
        if 'dlnPD_l1' in model_vars:
            if q == 1:
                # For first forecast, use actual lag from data
                if 'dlnPD_l1' in data.columns:
                    current_dlnpd = data.iloc[start_idx]['dlnPD_l1'] if pd.notna(data.iloc[start_idx]['dlnPD_l1']) else 0
                else:
                    current_dlnpd = 0
            # For subsequent forecasts, current_dlnpd is updated from previous iteration
        
        # Update mean-reverting term if present
        if 'mean_reverting' in model_vars:
            # Use the mean-reverting value from the target period (realized data)
            if 'mean_reverting' in data.columns:
                mean_rev_val = data.iloc[target_idx]['mean_reverting'] if pd.notna(data.iloc[target_idx]['mean_reverting']) else 0
            else:
                mean_rev_val = 0
        
        # Prepare input variables using realized MV data
        X_values = {}
        
        for orig_var in model_vars:
            if orig_var == 'const':
                X_values[orig_var] = 1.0
            elif orig_var == 'dlnPD_l1':
                X_values[orig_var] = current_dlnpd
            elif orig_var == 'mean_reverting':
                X_values[orig_var] = mean_rev_val
            else:
                # This is a macro variable - use realized value from target period
                if orig_var in data.columns:
                    val = data.iloc[target_idx][orig_var]
                    X_values[orig_var] = val if pd.notna(val) else 0.0
                else:
                    X_values[orig_var] = 0.0
        
        # Calculate predicted dlnPD
        predicted_dlnpd = sum(coeff_dict.get(var, 0) * X_values.get(var, 0) for var in model_vars)
        
        # Calculate predicted PD: exp(current_lnPD + dlnPD)
        new_lnpd = current_lnpd + predicted_dlnpd
        predicted_pd = np.exp(new_lnpd)
        
        # Get actual values for comparison
        actual_pd = data.iloc[target_idx]['cdsiedf5'] if pd.notna(data.iloc[target_idx]['cdsiedf5']) else np.nan
        actual_dlnpd = data.iloc[target_idx]['dlnPD'] if 'dlnPD' in data.columns and pd.notna(data.iloc[target_idx]['dlnPD']) else np.nan
        
        # Store results
        forecast_results.append({
            'yyyyqq': data.iloc[target_idx]['yyyyqq'],
            'quarter_ahead': q,
            'predicted_dlnPD': predicted_dlnpd,
            'predicted_PD': predicted_pd,
            'actual_PD': actual_pd,
            'actual_dlnPD': actual_dlnpd,
            'forecast_start_date': data.iloc[start_idx]['yyyyqq']
        })
        
        # Update current values for next iteration
        current_dlnpd = predicted_dlnpd
        current_lnpd = new_lnpd
    
    return pd.DataFrame(forecast_results)

def run_backtesting_for_country(country, data, model_info, horizon_quarters=DEFAULT_HORIZON):
    """
    Run backtesting for a specific country across all possible starting quarters.
    
    Args:
        country: Country code
        data: DataFrame with historical data
        model_info: Dictionary with model variables and coefficients
        horizon_quarters: Number of quarters to forecast ahead
    
    Returns:
        DataFrame with all backtesting results
    """
    # Filter data for the country and sort by date
    country_data = data[data['cinc'] == country].copy()
    country_data = country_data.sort_values('yyyyqq').reset_index(drop=True)
    
    # Filter to only periods with PD data
    country_data = country_data[country_data['cdsiedf5'].notna()]
    
    # Since we use pre-trained coefficients from step 7, we only need minimal data:
    # 1. At least MIN_LAG_QUARTERS for lag variables
    # 2. Enough future periods for the forecast horizon
    if len(country_data) < MIN_LAG_QUARTERS + horizon_quarters:
        print(f"  Insufficient data for {country}: {len(country_data)} quarters available, need {MIN_LAG_QUARTERS + horizon_quarters}")
        return pd.DataFrame()
    
    all_backtests = []
    
    # Run backtesting starting from when data is available (using pre-trained coefficients)
    # Start from MIN_LAG_QUARTERS to ensure we have lag variables available
    for start_idx in range(MIN_LAG_QUARTERS, len(country_data) - horizon_quarters):
        backtest_result = forecast_dlnpd_backtesting(
            country_data, model_info, start_idx, horizon_quarters
        )
        
        if not backtest_result.empty:
            backtest_result['country'] = country
            backtest_result['start_idx'] = start_idx
            all_backtests.append(backtest_result)
    
    if all_backtests:
        combined_results = pd.concat(all_backtests, ignore_index=True)
        print(f"  Generated {len(all_backtests)} backtests for {country} (horizon: {horizon_quarters}Q)")
        return combined_results
    else:
        print(f"  No valid backtests for {country}")
        return pd.DataFrame()

def calculate_backtesting_metrics(backtest_results):
    """Calculate performance metrics for backtesting results"""
    if backtest_results.empty:
        return {}
    
    # Remove NaN values
    valid_data = backtest_results.dropna(subset=['predicted_PD', 'actual_PD'])
    
    if len(valid_data) == 0:
        return {}
    
    # Overall metrics
    metrics = {}
    
    # PD level metrics
    metrics['n_forecasts'] = len(valid_data)
    metrics['pd_mae'] = np.mean(np.abs(valid_data['predicted_PD'] - valid_data['actual_PD']))
    metrics['pd_rmse'] = np.sqrt(np.mean((valid_data['predicted_PD'] - valid_data['actual_PD'])**2))
    metrics['pd_mape'] = np.mean(np.abs((valid_data['predicted_PD'] - valid_data['actual_PD']) / valid_data['actual_PD'])) * 100
    metrics['pd_correlation'] = valid_data['predicted_PD'].corr(valid_data['actual_PD'])
    
    # dlnPD metrics (if available)
    valid_dlnpd = valid_data.dropna(subset=['predicted_dlnPD', 'actual_dlnPD'])
    if not valid_dlnpd.empty:
        metrics['dlnpd_mae'] = np.mean(np.abs(valid_dlnpd['predicted_dlnPD'] - valid_dlnpd['actual_dlnPD']))
        metrics['dlnpd_rmse'] = np.sqrt(np.mean((valid_dlnpd['predicted_dlnPD'] - valid_dlnpd['actual_dlnPD'])**2))
        metrics['dlnpd_correlation'] = valid_dlnpd['predicted_dlnPD'].corr(valid_dlnpd['actual_dlnPD'])
    
    # Metrics by forecast horizon
    for q in valid_data['quarter_ahead'].unique():
        q_data = valid_data[valid_data['quarter_ahead'] == q]
        if not q_data.empty:
            metrics[f'pd_mape_q{q}'] = np.mean(np.abs((q_data['predicted_PD'] - q_data['actual_PD']) / q_data['actual_PD'])) * 100
            metrics[f'pd_corr_q{q}'] = q_data['predicted_PD'].corr(q_data['actual_PD'])
    
    return metrics

def calculate_model_level_summary(all_combinations_metrics):
    """Calculate model-level summary statistics across countries"""
    model_summary = []
    
    for combination_name, metrics_dict in all_combinations_metrics.items():
        if not metrics_dict:
            continue
            
        # Convert to DataFrame for easier aggregation
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Calculate averages across countries
        ovs_source, horizon_str = combination_name.rsplit('_', 1)
        horizon = int(horizon_str.replace('Q', ''))
        
        summary_row = {
            'combination': combination_name,
            'ovs_source': ovs_source,
            'horizon_quarters': horizon,
            'n_countries': len(metrics_df),
            'avg_n_forecasts': metrics_df['n_forecasts'].mean(),
            'avg_pd_mae': metrics_df['pd_mae'].mean(),
            'avg_pd_rmse': metrics_df['pd_rmse'].mean(),
            'avg_pd_mape': metrics_df['pd_mape'].mean(),
            'avg_pd_correlation': metrics_df['pd_correlation'].mean(),
            'median_pd_mape': metrics_df['pd_mape'].median(),
            'median_pd_correlation': metrics_df['pd_correlation'].median(),
            'std_pd_mape': metrics_df['pd_mape'].std(),
            'std_pd_correlation': metrics_df['pd_correlation'].std(),
            'min_pd_mape': metrics_df['pd_mape'].min(),
            'max_pd_mape': metrics_df['pd_mape'].max(),
            'min_pd_correlation': metrics_df['pd_correlation'].min(),
            'max_pd_correlation': metrics_df['pd_correlation'].max()
        }
        
        # Add dlnPD metrics if available
        if 'dlnpd_mae' in metrics_df.columns:
            summary_row.update({
                'avg_dlnpd_mae': metrics_df['dlnpd_mae'].mean(),
                'avg_dlnpd_rmse': metrics_df['dlnpd_rmse'].mean(),
                'avg_dlnpd_correlation': metrics_df['dlnpd_correlation'].mean(),
                'median_dlnpd_correlation': metrics_df['dlnpd_correlation'].median()
            })
        
        # Add horizon-specific metrics
        for q in [1, 2, 3, 4]:  # Common quarters
            mape_col = f'pd_mape_q{q}'
            corr_col = f'pd_corr_q{q}'
            if mape_col in metrics_df.columns:
                summary_row[f'avg_pd_mape_q{q}'] = metrics_df[mape_col].mean()
            if corr_col in metrics_df.columns:
                summary_row[f'avg_pd_corr_q{q}'] = metrics_df[corr_col].mean()
        
        model_summary.append(summary_row)
    
    return pd.DataFrame(model_summary)

def plot_backtesting_results(country, backtest_results, model_info, plots_dir=None, combination_name=""):
    """Plot backtesting results for a country"""
    if plots_dir is None:
        plots_dir = Path('Output/8.backtesting_analysis/plots')
    
    if backtest_results.empty:
        return
    
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
    
    # Filter valid data
    valid_data = backtest_results.dropna(subset=['predicted_PD', 'actual_PD'])
    
    if valid_data.empty:
        plt.rcdefaults()
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Predicted vs Actual PD scatter plot (convert to percentage)
    ax1 = axes[0, 0]
    ax1.scatter(valid_data['actual_PD'] * 100, valid_data['predicted_PD'] * 100, alpha=0.6, s=20)
    
    # Add perfect prediction line
    min_pd = min(valid_data['actual_PD'].min(), valid_data['predicted_PD'].min()) * 100
    max_pd = max(valid_data['actual_PD'].max(), valid_data['predicted_PD'].max()) * 100
    ax1.plot([min_pd, max_pd], [min_pd, max_pd], 'r--', label='Perfect Prediction', linewidth=2)
    
    ax1.set_xlabel('Actual PD (%)', fontsize=14)
    ax1.set_ylabel('Predicted PD (%)', fontsize=14)
    ax1.set_title('Predicted vs Actual PD', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Error distribution by forecast horizon
    ax2 = axes[0, 1]
    horizons = sorted(valid_data['quarter_ahead'].unique())
    for q in horizons:
        q_data = valid_data[valid_data['quarter_ahead'] == q]
        errors = ((q_data['predicted_PD'] - q_data['actual_PD']) / q_data['actual_PD']) * 100
        ax2.hist(errors, bins=20, alpha=0.7, label=f'{q}Q ahead')
    
    ax2.set_xlabel('Prediction Error (%)', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('Error Distribution by Forecast Horizon', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Time series of prediction accuracy
    ax3 = axes[1, 0]
    # Calculate rolling MAPE for each forecast horizon
    for q in horizons:
        q_data = valid_data[valid_data['quarter_ahead'] == q].sort_values('yyyyqq')
        if not q_data.empty:
            mape_values = np.abs((q_data['predicted_PD'] - q_data['actual_PD']) / q_data['actual_PD']) * 100
            ax3.plot(q_data['yyyyqq'], mape_values, label=f'{q}Q ahead', alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Date', fontsize=14)
    ax3.set_ylabel('MAPE (%)', fontsize=14)
    ax3.set_title('Prediction Accuracy Over Time', fontsize=16)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: Cumulative error statistics
    ax4 = axes[1, 1]
    horizon_stats = []
    for q in horizons:
        q_data = valid_data[valid_data['quarter_ahead'] == q]
        if not q_data.empty:
            mape = np.mean(np.abs((q_data['predicted_PD'] - q_data['actual_PD']) / q_data['actual_PD'])) * 100
            corr = q_data['predicted_PD'].corr(q_data['actual_PD'])
            horizon_stats.append({'Quarter': q, 'MAPE': mape, 'Correlation': corr})
    
    if horizon_stats:
        stats_df = pd.DataFrame(horizon_stats)
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(stats_df['Quarter'], stats_df['MAPE'], alpha=0.7, color='orange', label='MAPE (%)')
        line1 = ax4_twin.plot(stats_df['Quarter'], stats_df['Correlation'], 'ro-', label='Correlation', linewidth=2, markersize=8)
        
        ax4.set_xlabel('Forecast Horizon (Quarters)', fontsize=14)
        ax4.set_ylabel('MAPE (%)', color='orange', fontsize=14)
        ax4_twin.set_ylabel('Correlation', color='red', fontsize=14)
        ax4.set_title('Performance by Forecast Horizon', fontsize=16)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4_twin.tick_params(axis='both', which='major', labelsize=12)
        
        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    # Main title with model info (remove 'Model:' prefix and show R-squared as percentage)
    ovs_source = model_info.get('ovs_source', 'advanced')
    model_desc = get_model_description_for_plotting(country, ovs_source)
    
    fig.suptitle(f'Backtesting Results: {country}\n'
                f'{model_desc}\n'
                f'Adjusted R-squared: {model_info["adj_r2"]*100:.2f}%', fontsize=18)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save plot
    plt.savefig(plots_dir / f'backtesting_{country}_{combination_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib rcParams to default
    plt.rcdefaults()
    
    print(f"  Saved backtesting plot for {country}")

def plot_cumulative_backtesting_results(country, backtest_results, model_info, plots_dir=None, combination_name=""):
    """Plot cumulative backtesting results showing sum of forecasted vs realized PD over N quarters"""
    if plots_dir is None:
        plots_dir = Path('Output/8.backtesting_analysis/plots')
    
    if backtest_results.empty:
        return
    
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
    
    # Filter valid data
    valid_data = backtest_results.dropna(subset=['predicted_PD', 'actual_PD'])
    
    if valid_data.empty:
        plt.rcdefaults()
        return
    
    # Get unique forecast start dates
    forecast_starts = sorted(valid_data['forecast_start_date'].unique())
    
    # Aggregate cumulative PD for each forecast start date
    cumulative_results = []
    
    for start_date in forecast_starts:
        # Get all forecasts for this start date
        start_data = valid_data[valid_data['forecast_start_date'] == start_date].copy()
        
        if start_data.empty:
            continue
        
        # Sort by quarter ahead to ensure proper cumulation
        start_data = start_data.sort_values('quarter_ahead')
        
        # Calculate cumulative sum of PD across all quarters in the horizon
        cumulative_predicted = start_data['predicted_PD'].sum()
        cumulative_actual = start_data['actual_PD'].sum()
        
        # Get the maximum horizon for this forecast (for display purposes)
        max_horizon = start_data['quarter_ahead'].max()
        
        cumulative_results.append({
            'forecast_start_date': start_date,
            'cumulative_predicted_PD': cumulative_predicted,
            'cumulative_actual_PD': cumulative_actual,
            'max_horizon': max_horizon,
            'n_quarters': len(start_data)
        })
    
    if not cumulative_results:
        plt.rcdefaults()
        return
    
    # Convert to DataFrame for easier plotting
    cum_df = pd.DataFrame(cumulative_results)
    cum_df = cum_df.sort_values('forecast_start_date')
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot forecasted vs realized cumulative PD over time (convert to percentage)
    ax.plot(cum_df['forecast_start_date'], cum_df['cumulative_predicted_PD'] * 100, 
            color='blue', linestyle='-', linewidth=3, alpha=0.8,
            label='Forecasted Cumulative PD', marker='o', markersize=6)
    
    ax.plot(cum_df['forecast_start_date'], cum_df['cumulative_actual_PD'] * 100, 
            color='red', linestyle='--', linewidth=3, alpha=0.8,
            label='Realized Cumulative PD', marker='s', markersize=6)
    
    # Calculate and show overall correlation and MAPE
    corr = cum_df['cumulative_predicted_PD'].corr(cum_df['cumulative_actual_PD'])
    mape = np.mean(np.abs((cum_df['cumulative_predicted_PD'] - cum_df['cumulative_actual_PD']) / cum_df['cumulative_actual_PD'])) * 100
    
    # Get horizon info for title
    if len(cum_df) > 0:
        typical_horizon = cum_df['max_horizon'].mode().iloc[0] if not cum_df['max_horizon'].mode().empty else cum_df['max_horizon'].iloc[0]
        typical_n_quarters = cum_df['n_quarters'].mode().iloc[0] if not cum_df['n_quarters'].mode().empty else cum_df['n_quarters'].iloc[0]
    else:
        typical_horizon = 0
        typical_n_quarters = 0
    
    # Main title with model info (remove 'Model:' prefix and show R-squared as percentage)
    ovs_source = model_info.get('ovs_source', 'advanced')
    model_desc = get_model_description_for_plotting(country, ovs_source)
    
    fig.suptitle(f'{country} - {model_desc}\n'
                f'Adjusted R-squared: {model_info["adj_r2"]*100:.2f}%', fontsize=18, y=0.98)
    
    ax.set_title(f'Cumulative {int(typical_horizon)}Q PD Forecasts vs Realizations\n'
                f'Sum of {int(typical_n_quarters)} quarterly PDs | Correlation: {corr:.3f}, MAPE: {mape:.1f}%', 
                fontsize=16, pad=30)
    ax.set_xlabel('Forecast Start Date', fontsize=14)
    ax.set_ylabel('Cumulative PD (% - Sum of Quarterly PDs)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add some statistics text (convert to percentage)
    stats_text = f'Forecasts: {len(cum_df)}\n' \
                f'Avg Predicted: {cum_df["cumulative_predicted_PD"].mean()*100:.4f}%\n' \
                f'Avg Realized: {cum_df["cumulative_actual_PD"].mean()*100:.4f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    
    # Save plot
    plt.savefig(plots_dir / f'cumulative_backtesting_{country}_{combination_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset matplotlib rcParams to default
    plt.rcdefaults()
    
    print(f"  Saved cumulative backtesting plot for {country}")

def main():
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    
    # Initialize master summary for all combinations
    all_combinations_summary = []
    all_combinations_metrics = {}
    
    # Process all combinations of OVS sources and horizons
    for ovs_source in OVS_SOURCES.keys():
        for horizon in BACKTESTING_HORIZONS:
            combination_name = f"{ovs_source}_{horizon}Q"
            print(f"\nProcessing: {combination_name}")
            
            try:
                ovs_models = load_ovs_results(ovs_source, 'has_no_lags')
                
                # Skip if no has_no_lags models found
                if not ovs_models:
                    continue
                
                all_backtest_results = {}
                backtest_metrics = {}
                
                # Process each country
                for country, model_info in ovs_models.items():
                    try:
                        # Create country-specific output directory
                        country_output_dir = output_dir / country
                        country_output_dir.mkdir(parents=True, exist_ok=True)
                        country_plots_dir = country_output_dir / 'plots'
                        country_plots_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Run backtesting for this country
                        backtest_results = run_backtesting_for_country(
                            country, data, model_info, horizon
                        )
                        
                        if not backtest_results.empty:
                            # Save backtesting results
                            backtest_results.to_csv(
                                country_output_dir / f'backtest_results_{combination_name}.csv', 
                                index=False
                            )
                            
                            # Calculate metrics
                            metrics = calculate_backtesting_metrics(backtest_results)
                            if metrics:
                                backtest_metrics[country] = metrics
                            
                            # Create plots - only keep cumulative backtesting time series plot
                            # plot_backtesting_results(
                            #     country, backtest_results, model_info, 
                            #     country_plots_dir, combination_name
                            # )
                            
                            # Create cumulative backtesting plots
                            plot_cumulative_backtesting_results(
                                country, backtest_results, model_info, 
                                country_plots_dir, combination_name
                            )
                            
                            # Copy to comparison directory - only cumulative backtesting chart
                            # source_chart = country_plots_dir / f'backtesting_{country}_{combination_name}.png'
                            # target_chart = comparison_plots_dir / f'backtesting_{country}_{combination_name}.png'
                            # if source_chart.exists():
                            #     shutil.copy2(source_chart, target_chart)
                            
                            # Copy cumulative backtesting chart to comparison directory
                            source_cumulative_chart = country_plots_dir / f'cumulative_backtesting_{country}_{combination_name}.png'
                            target_cumulative_chart = comparison_plots_dir / f'cumulative_backtesting_{country}_{combination_name}.png'
                            if source_cumulative_chart.exists():
                                shutil.copy2(source_cumulative_chart, target_cumulative_chart)
                            
                            all_backtest_results[country] = backtest_results
                            
                    except Exception as e:
                        print(f"Error processing {country}: {str(e)}")
                        continue
                
                # Store metrics for this combination
                all_combinations_metrics[combination_name] = backtest_metrics
                
                # Add to master summary
                for country, backtest_df in all_backtest_results.items():
                    if not backtest_df.empty:
                        country_metrics = backtest_metrics.get(country, {})
                        
                        all_combinations_summary.append({
                            'combination': combination_name,
                            'ovs_source': ovs_source,
                            'horizon_quarters': horizon,
                            'country': country,
                            'n_backtests': len(backtest_df['start_idx'].unique()),
                            'n_forecasts': len(backtest_df),
                            'model_adj_r2': ovs_models[country]['adj_r2'],
                            'pd_mape': country_metrics.get('pd_mape', np.nan),
                            'pd_correlation': country_metrics.get('pd_correlation', np.nan),
                            'pd_rmse': country_metrics.get('pd_rmse', np.nan),
                            'dlnpd_rmse': country_metrics.get('dlnpd_rmse', np.nan),
                            'dlnpd_correlation': country_metrics.get('dlnpd_correlation', np.nan)
                        })
                
                # Print summary statistics for this combination
                if backtest_metrics:
                    metrics_df = pd.DataFrame.from_dict(backtest_metrics, orient='index')
                    print(f"\n{combination_name} Performance Summary:")
                    print(f"  Average PD MAPE: {metrics_df['pd_mape'].mean():.1f}%")
                    print(f"  Average PD Correlation: {metrics_df['pd_correlation'].mean():.3f}")
                    print(f"  Average PD RMSE: {metrics_df['pd_rmse'].mean():.4f}")
                
                print(f"{combination_name} complete: {len(all_backtest_results)} countries processed")
                
            except Exception as e:
                print(f"Error processing combination {combination_name}: {str(e)}")
                continue
    
    # Save master summary and metrics files
    if all_combinations_summary:
        master_summary_df = pd.DataFrame(all_combinations_summary)
        master_summary_df.to_csv(output_dir / 'master_backtesting_summary_all_combinations.csv', index=False)
    
    # Save master metrics summary
    if all_combinations_metrics:
        master_metrics_data = []
        for combination_name, metrics_dict in all_combinations_metrics.items():
            ovs_source, horizon_str = combination_name.rsplit('_', 1)
            horizon = int(horizon_str.replace('Q', ''))
            
            for country, metrics in metrics_dict.items():
                metrics_record = {
                    'combination': combination_name,
                    'ovs_source': ovs_source,
                    'horizon_quarters': horizon,
                    'country': country,
                    **metrics
                }
                master_metrics_data.append(metrics_record)
        
        master_metrics_df = pd.DataFrame(master_metrics_data)
        master_metrics_df.to_csv(output_dir / 'master_backtesting_metrics_all_combinations.csv', index=False)
        
        # Calculate and save model-level summary
        model_summary_df = calculate_model_level_summary(all_combinations_metrics)
        if not model_summary_df.empty:
            model_summary_df.to_csv(output_dir / 'model_level_backtesting_summary.csv', index=False)
            
            # Print model-level summary
            print(f"\nMODEL-LEVEL BACKTESTING SUMMARY:")
            print(f"{'='*60}")
            for _, row in model_summary_df.iterrows():
                print(f"{row['combination']}: {row['n_countries']} countries, "
                      f"Avg MAPE: {row['avg_pd_mape']:.1f}%, "
                      f"Avg Correlation: {row['avg_pd_correlation']:.3f}")
            print(f"{'='*60}")
    
    # Create country-specific summary files
    if all_combinations_summary:
        master_df = pd.DataFrame(all_combinations_summary)
        countries = master_df['country'].unique()
        
        for country in countries:
            country_data = master_df[master_df['country'] == country]
            country_output_dir = output_dir / country
            country_data.to_csv(country_output_dir / f'{country}_backtesting_summary.csv', index=False)
            
            # Also create country-specific metrics file if available
            if all_combinations_metrics:
                country_metrics_data = []
                for combination_name, metrics_dict in all_combinations_metrics.items():
                    if country in metrics_dict:
                        ovs_source, horizon_str = combination_name.rsplit('_', 1)
                        horizon = int(horizon_str.replace('Q', ''))
                        
                        metrics_record = {
                            'combination': combination_name,
                            'ovs_source': ovs_source,
                            'horizon_quarters': horizon,
                            **metrics_dict[country]
                        }
                        country_metrics_data.append(metrics_record)
                
                if country_metrics_data:
                    country_metrics_df = pd.DataFrame(country_metrics_data)
                    country_metrics_df.to_csv(country_output_dir / f'{country}_backtesting_metrics.csv', index=False)
    
    print(f"\nBACKTESTING ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated combinations: {len(OVS_SOURCES)} OVS sources × {len(BACKTESTING_HORIZONS)} horizons")
    print(f"Forecast horizons: {BACKTESTING_HORIZONS} quarters")
    print(f"Master files: summary and metrics CSVs created")
    print(f"Comparison charts: {comparison_plots_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 