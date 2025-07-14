import pandas as pd
import numpy as np
from pandas.tseries.offsets import QuarterEnd
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import X-13ARIMA-SEATS, fall back to STL if not available
try:
    from statsmodels.tsa.x13 import x13_arima_analysis
    X13_AVAILABLE = True
    print("X-13ARIMA-SEATS is available")
except ImportError:
    X13_AVAILABLE = False
    print("X-13ARIMA-SEATS not available")

# Import STL decomposition as fallback
try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
    print("STL decomposition is available")
except ImportError:
    STL_AVAILABLE = False
    print("STL decomposition not available - using basic seasonal_decompose")

# Import basic seasonal decomposition as final fallback
from statsmodels.tsa.seasonal import seasonal_decompose

def get_scenario_mnemonics(baseline_mnemonic):
    """Generate scenario mnemonics for a given baseline mnemonic."""
    scenarios = ['_S1', '_S3', '_S4']
    return [f"{baseline_mnemonic}{scen}" for scen in scenarios]

def convert_to_date_string(x):
    """Convert float date (e.g. 2025.25) to proper date string."""
    try:
        x = float(x)
        year = int(x)
        quarter = int(round((x - year) * 4 + 1))
        month = quarter * 3
        return f"{year}-{month:02d}-01"
    except Exception as e:
        print(f"Failed to convert date {x}: {str(e)}")
        return None

def apply_x13_to_pd_data(data, pd_col='cdsiedf5', country_col='cinc', min_coverage=0.90):
    """
    Apply X-13ARIMA-SEATS seasonal adjustment to quarterly PD data.
    Falls back to STL decomposition if X-13 is not available or fails.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing PD values with yyyyqq and country columns
    pd_col : str
        Column name for PD values (default: 'cdsiedf5')
    country_col : str
        Column name for country identifier (default: 'cinc')
    min_coverage : float
        Minimum data coverage required to attempt seasonal adjustment (default: 0.90)
    
    Returns:
    --------
    pd.DataFrame
        Data with additional columns for seasonally adjusted PD
    """
    print(f"\\nApplying X-13 seasonal adjustment to {pd_col}...")
    
    # Create output directory for plots
    output_dir = Path('Output/3.pd_seasonal_adjustment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result columns
    data[f'{pd_col}_sa'] = data[pd_col].copy()  # Seasonally adjusted PD
    data[f'{pd_col}_seasonal'] = np.nan  # Seasonal component
    data[f'{pd_col}_sa_method'] = 'none'  # Method used
    
    # Track adjustment summary
    adjustment_summary = []
    
    # Process each country
    countries = data[country_col].unique()
    print(f"Processing {len(countries)} countries...")
    
    for country in countries:
        country_data = data[data[country_col] == country].copy()
        
        # Filter to PD data period only (where PD exists)
        pd_data = country_data[country_data[pd_col].notna()].copy()
        
        if len(pd_data) < 8:  # Need at least 2 years of quarterly data
            print(f"  {country}: Insufficient PD data ({len(pd_data)} quarters)")
            adjustment_summary.append({'country': country, 'method': 'none', 'reason': 'insufficient_data'})
            continue
        
        # Calculate coverage within PD period
        pd_period_start = pd_data['yyyyqq'].min()
        pd_period_end = pd_data['yyyyqq'].max()
        expected_quarters = len(pd.date_range(pd_period_start, pd_period_end, freq='Q'))
        actual_quarters = len(pd_data)
        coverage = actual_quarters / expected_quarters
        
        if coverage < min_coverage:
            print(f"  {country}: Low PD coverage ({coverage:.1%}) in period {pd_period_start.strftime('%Y-%m')} to {pd_period_end.strftime('%Y-%m')}")
            adjustment_summary.append({'country': country, 'method': 'none', 'reason': 'low_coverage'})
            continue
        
        # Prepare time series for seasonal adjustment
        ts_data = pd_data.set_index('yyyyqq')[pd_col].sort_index()
        
        # Fill small gaps (≤2 consecutive quarters) with interpolation
        if ts_data.isna().any():
            ts_data = ts_data.interpolate(method='linear', limit=2)
        
        method_used = 'none'
        seasonal_component = None
        adjusted_series = None
        
        # Try X-13ARIMA-SEATS first
        if X13_AVAILABLE and len(ts_data.dropna()) >= 12:
            try:
                print(f"  {country}: Attempting X-13ARIMA-SEATS...")
                
                # X-13 requires no missing values
                ts_clean = ts_data.dropna()
                if len(ts_clean) >= 12:
                    result = x13_arima_analysis(
                        ts_clean,
                        outlier=True,
                        trading=False,
                        forecast_years=0
                    )
                    
                    seasonal_component = result.seasadj - result.observed
                    adjusted_series = result.seasadj
                    method_used = 'x13'
                    print(f"  {country}: X-13 successful")
                
            except Exception as e:
                print(f"  {country}: X-13 failed ({str(e)[:50]}...) - using original time series")
                # When X13 fails, use original time series instead of other methods
                adjusted_series = ts_data.dropna()
                seasonal_component = pd.Series(0, index=adjusted_series.index)  # No seasonal adjustment
                method_used = 'original'
        else:
            if not X13_AVAILABLE:
                print(f"  {country}: X-13 not available - using original time series")
            else:
                print(f"  {country}: Insufficient data for X-13 (need ≥12 obs, have {len(ts_data.dropna())}) - using original time series")
            # Use original time series when X13 is not available or insufficient data
            adjusted_series = ts_data.dropna()
            seasonal_component = pd.Series(0, index=adjusted_series.index)  # No seasonal adjustment
            method_used = 'original'
        
        # Store results if successful
        if method_used != 'none' and adjusted_series is not None:
            # Map back to original data
            for idx in adjusted_series.index:
                mask = (data[country_col] == country) & (data['yyyyqq'] == idx)
                if mask.any():
                    data.loc[mask, f'{pd_col}_sa'] = adjusted_series[idx]
                    data.loc[mask, f'{pd_col}_sa_method'] = method_used
                    
                    if seasonal_component is not None and idx in seasonal_component.index:
                        data.loc[mask, f'{pd_col}_seasonal'] = seasonal_component[idx]
            
            adjustment_summary.append({
                'country': country, 
                'method': method_used, 
                'reason': 'success' if method_used == 'x13' else 'x13_failed_or_unavailable',
                'periods_adjusted': len(adjusted_series)
            })
        else:
            adjustment_summary.append({'country': country, 'method': 'none', 'reason': 'no_data'})
    
    # Print summary
    summary_df = pd.DataFrame(adjustment_summary)
    method_counts = summary_df['method'].value_counts()
    print(f"\\nPD Seasonal Adjustment Summary:")
    print(f"  X-13ARIMA-SEATS: {method_counts.get('x13', 0)} countries")
    print(f"  Original time series (X-13 failed/unavailable): {method_counts.get('original', 0)} countries")
    print(f"  No data: {method_counts.get('none', 0)} countries")
    
    # Save adjustment summary
    summary_df.to_csv(output_dir / 'pd_seasonal_adjustment_summary.csv', index=False)
    
    # Count successful adjustments
    adjusted_countries = summary_df[summary_df['method'] != 'none']
    x13_success = summary_df[summary_df['method'] == 'x13']
    original_used = summary_df[summary_df['method'] == 'original']
    
    print(f"\\nProcessed PD data for {len(adjusted_countries)} countries:")
    print(f"  X-13 seasonal adjustment applied: {len(x13_success)} countries")
    print(f"  Original time series used: {len(original_used)} countries")
    
    return data

def create_pd_seasonal_adjustment_plots(data, output_dir):
    """Create comprehensive plots for PD seasonal adjustment results"""
    
    # Filter to countries with seasonal adjustment applied
    sa_countries = data[data['cdsiedf5_sa_method'] != 'none']['cinc'].unique()
    
    if len(sa_countries) == 0:
        print("No countries had seasonal adjustment applied - skipping PD comparison plots")
        return
    
    print(f"Creating PD seasonal adjustment plots for {len(sa_countries)} countries...")
    
    # 1. Time series comparison for ALL countries (multi-page plots)
    countries_per_page = 12  # 4x3 grid per page
    num_pages = (len(sa_countries) + countries_per_page - 1) // countries_per_page
    
    for page in range(num_pages):
        start_idx = page * countries_per_page
        end_idx = min((page + 1) * countries_per_page, len(sa_countries))
        page_countries = sa_countries[start_idx:end_idx]
        
        # Create subplot grid
        rows = 4
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, country in enumerate(page_countries):
            if i >= len(axes):
                break
                
            country_data = data[data['cinc'] == country].sort_values('yyyyqq')
            
            axes[i].plot(country_data['yyyyqq'], country_data['cdsiedf5'], 
                        label='Original', alpha=0.7, linewidth=2)
            
            # Get method used to determine appropriate label
            method = country_data['cdsiedf5_sa_method'].iloc[0] if len(country_data) > 0 else 'unknown'
            method = str(method) if method is not None else 'unknown'
            
            if method == 'x13':
                sa_label = 'X-13 Seasonally Adjusted'
            elif method == 'original':
                sa_label = 'Original (X-13 Failed)'
            else:
                sa_label = 'Processed'
                
            axes[i].plot(country_data['yyyyqq'], country_data['cdsiedf5_sa'], 
                        label=sa_label, alpha=0.7, linewidth=2)
            
            # Add seasonal component only if actually seasonally adjusted
            if method == 'x13' and 'cdsiedf5_seasonal' in country_data.columns:
                seasonal_data = country_data['cdsiedf5_seasonal'].dropna()
                if len(seasonal_data) > 0:
                    # Only plot seasonal component where it exists (aligned with country_data)
                    seasonal_mask = country_data['cdsiedf5_seasonal'].notna()
                    if seasonal_mask.sum() > 0:
                        seasonal_scaled = country_data.loc[seasonal_mask, 'cdsiedf5_seasonal'] * 10
                        axes[i].plot(country_data.loc[seasonal_mask, 'yyyyqq'], seasonal_scaled, 
                                    label='Seasonal (×10)', alpha=0.5, linestyle='--')
            
            axes[i].set_title(f'{country} - PD Processing ({method.upper()})')
            axes[i].set_ylabel('5Y CDS-Implied EDF', fontsize=8)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45, labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)
        
        # Hide unused subplots
        for i in range(len(page_countries), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'pd_seasonal_adjustment_timeseries_page_{page+1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Created {num_pages} time series plot pages")
    
    # 2. Summary plot with first 6 countries (for quick overview)
    sample_countries = sa_countries[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, country in enumerate(sample_countries):
        if i >= 6:
            break
            
        country_data = data[data['cinc'] == country].sort_values('yyyyqq')
        
        axes[i].plot(country_data['yyyyqq'], country_data['cdsiedf5'], 
                    label='Original', alpha=0.7, linewidth=2)
        axes[i].plot(country_data['yyyyqq'], country_data['cdsiedf5_sa'], 
                    label='Seasonally Adjusted', alpha=0.7, linewidth=2)
        
        # Add seasonal component (scaled for visibility)
        if 'cdsiedf5_seasonal' in country_data.columns:
            seasonal_data = country_data['cdsiedf5_seasonal'].dropna()
            if len(seasonal_data) > 0:
                # Only plot seasonal component where it exists (aligned with country_data)
                seasonal_mask = country_data['cdsiedf5_seasonal'].notna()
                if seasonal_mask.sum() > 0:
                    seasonal_scaled = country_data.loc[seasonal_mask, 'cdsiedf5_seasonal'] * 10
                    axes[i].plot(country_data.loc[seasonal_mask, 'yyyyqq'], seasonal_scaled, 
                                label='Seasonal (×10)', alpha=0.5, linestyle='--')
        
        method = country_data['cdsiedf5_sa_method'].iloc[0] if len(country_data) > 0 else 'unknown'
        method = str(method) if method is not None else 'unknown'
        axes[i].set_title(f'{country} - PD Seasonal Adjustment ({method.upper()})')
        axes[i].set_ylabel('5Y CDS-Implied EDF')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pd_seasonal_adjustment_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution comparison
    plt.figure(figsize=(15, 10))
    
    # Original vs SA PD distributions
    plt.subplot(2, 3, 1)
    plt.hist(data['cdsiedf5'].dropna(), bins=50, alpha=0.7, label='Original', density=True)
    plt.hist(data['cdsiedf5_sa'].dropna(), bins=50, alpha=0.7, label='Seasonally Adjusted', density=True)
    plt.xlabel('5Y CDS-Implied EDF')
    plt.ylabel('Density')
    plt.title('PD Distribution Comparison')
    plt.legend()
    plt.yscale('log')
    
    # dlnPD distributions (if available)
    plt.subplot(2, 3, 2)
    if 'dlnPD' in data.columns and 'dlnPD_original' in data.columns:
        plt.hist(data['dlnPD_original'].dropna(), bins=50, alpha=0.7, label='Original dlnPD', density=True)
        plt.hist(data['dlnPD'].dropna(), bins=50, alpha=0.7, label='SA dlnPD', density=True)
        plt.xlabel('dlnPD')
        plt.ylabel('Density')
        plt.title('dlnPD Distribution Comparison')
        plt.legend()
    
    # Seasonal component distribution
    plt.subplot(2, 3, 3)
    seasonal_data = data['cdsiedf5_seasonal'].dropna()
    if len(seasonal_data) > 0:
        plt.hist(seasonal_data, bins=50, alpha=0.7, color='orange')
        plt.xlabel('Seasonal Component')
        plt.ylabel('Frequency')
        plt.title('Seasonal Component Distribution')
    
    # Method usage pie chart
    plt.subplot(2, 3, 4)
    method_counts = data.groupby('cinc')['cdsiedf5_sa_method'].first().value_counts()
    plt.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    plt.title(f'Seasonal Adjustment Methods Used\\n({len(sa_countries)} countries)')
    
    # Outlier impact comparison
    plt.subplot(2, 3, 5)
    # Calculate outlier percentages before and after SA
    original_outliers = (data['cdsiedf5'] > data['cdsiedf5'].quantile(0.99)).sum()
    sa_outliers = (data['cdsiedf5_sa'] > data['cdsiedf5_sa'].quantile(0.99)).sum()
    
    plt.bar(['Original', 'Seasonally Adjusted'], [original_outliers, sa_outliers])
    plt.ylabel('Number of Extreme Values (>99th percentile)')
    plt.title('Outlier Reduction')
    
    # Correlation between original and SA
    plt.subplot(2, 3, 6)
    valid_mask = data['cdsiedf5'].notna() & data['cdsiedf5_sa'].notna()
    if valid_mask.sum() > 0:
        correlation = data.loc[valid_mask, 'cdsiedf5'].corr(data.loc[valid_mask, 'cdsiedf5_sa'])
        plt.scatter(data.loc[valid_mask, 'cdsiedf5'], data.loc[valid_mask, 'cdsiedf5_sa'], 
                   alpha=0.1, s=1)
        plt.xlabel('Original PD')
        plt.ylabel('Seasonally Adjusted PD')
        plt.title(f'Original vs SA PD\\n(Correlation: {correlation:.3f})')
        
        # Add diagonal line
        min_val = min(data.loc[valid_mask, 'cdsiedf5'].min(), data.loc[valid_mask, 'cdsiedf5_sa'].min())
        max_val = max(data.loc[valid_mask, 'cdsiedf5'].max(), data.loc[valid_mask, 'cdsiedf5_sa'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pd_seasonal_adjustment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Country-specific impact summary
    impact_summary = []
    for country in sa_countries:
        country_data = data[data['cinc'] == country]
        
        if len(country_data) > 0:
            original_std = country_data['cdsiedf5'].std()
            sa_std = country_data['cdsiedf5_sa'].std()
            
            if 'dlnPD' in country_data.columns and 'dlnPD_original' in country_data.columns:
                original_dlnpd_std = country_data['dlnPD_original'].std()
                sa_dlnpd_std = country_data['dlnPD'].std()
            else:
                original_dlnpd_std = np.nan
                sa_dlnpd_std = np.nan
            
            method = country_data['cdsiedf5_sa_method'].iloc[0]
            
            impact_summary.append({
                'country': country,
                'method': method,
                'original_pd_std': original_std,
                'sa_pd_std': sa_std,
                'pd_volatility_change': (sa_std - original_std) / original_std * 100 if original_std > 0 else np.nan,
                'original_dlnpd_std': original_dlnpd_std,
                'sa_dlnpd_std': sa_dlnpd_std,
                'dlnpd_volatility_change': (sa_dlnpd_std - original_dlnpd_std) / original_dlnpd_std * 100 if not np.isnan(original_dlnpd_std) and original_dlnpd_std > 0 else np.nan
            })
    
    impact_df = pd.DataFrame(impact_summary)
    impact_df.to_csv(output_dir / 'pd_seasonal_adjustment_impact.csv', index=False)
    
    print(f"  Created comprehensive analysis plots and impact summary for all {len(sa_countries)} countries")
    print(f"  Files created:")
    print(f"    - {num_pages} detailed time series pages: pd_seasonal_adjustment_timeseries_page_*.png")
    print(f"    - Summary plot: pd_seasonal_adjustment_summary.png") 
    print(f"    - Analysis plot: pd_seasonal_adjustment_analysis.png")
    print(f"    - Impact CSV: pd_seasonal_adjustment_impact.csv")

def main():
    try:
        # Load sovereign PD data
        print("Loading sovereign PD data...")
        sov_pd = pd.read_csv('Output/1.sov_use_052025.csv')
        print(f"Unique countries in PD data: {sorted(sov_pd['cinc'].unique())}")
        
        sov_pd['price_date'] = pd.to_datetime(sov_pd['price_date'])
        sov_pd['yyyyqq'] = sov_pd['price_date'] + QuarterEnd(0)
        print(f"Date range in PD data: {sov_pd['yyyyqq'].min()} to {sov_pd['yyyyqq'].max()}")

        # Convert to quarterly frequency by taking average
        print("\nConverting PD to quarterly frequency...")
        sov_pd_q = sov_pd.groupby(['pid', 'entityName', 'cinc', 'region', 'yyyyqq'])[['cdsiedf5']].mean().reset_index()
        print(f"PD value range: {sov_pd_q['cdsiedf5'].min():.6f} to {sov_pd_q['cdsiedf5'].max():.6f}")
        
        # Apply X-13 seasonal adjustment to quarterly PD data
        print("\n" + "="*60)
        print("APPLYING X-13 SEASONAL ADJUSTMENT TO PD DATA")
        print("="*60)
        sov_pd_q = apply_x13_to_pd_data(sov_pd_q, pd_col='cdsiedf5', country_col='cinc')
        
        # Use seasonally adjusted PD for subsequent analysis
        # Create a column that uses SA PD where available, original PD otherwise
        sov_pd_q['cdsiedf5_final'] = sov_pd_q['cdsiedf5_sa'].fillna(sov_pd_q['cdsiedf5'])
        
        print(f"\nFinal PD data summary:")
        print(f"  Original PD observations: {sov_pd_q['cdsiedf5'].notna().sum()}")
        print(f"  Seasonally adjusted observations: {(sov_pd_q['cdsiedf5_sa_method'] != 'none').sum()}")
        print(f"  Using SA PD for analysis: {(sov_pd_q['cdsiedf5_sa_method'] != 'none').sum()}")
        print(f"  Using original PD for analysis: {(sov_pd_q['cdsiedf5_sa_method'] == 'none').sum()}")
        
        # Replace SCG with SRB for Serbia
        print("\nReplacing SCG with SRB for Serbia...")
        sov_pd_q['cinc'] = sov_pd_q['cinc'].replace('SCG', 'SRB')
        print("Updated country codes:", sorted(sov_pd_q['cinc'].unique()))
        
        # Load sovereign mnemonics mapping
        print("\nLoading sovereign mnemonics mapping...")
        sov_mnemonics = pd.read_csv('Output/2.sov_mv_mnemonics.csv')
        
        # Melt the mnemonics mapping to long format
        print("\nConverting mnemonics mapping to long format...")
        mnemonics_long = pd.melt(
            sov_mnemonics,
            id_vars=['cinc', 'entityName'],
            var_name='variable',
            value_name='ma.ticker'
        )
        mnemonics_long = mnemonics_long.dropna(subset=['ma.ticker'])
        print(f"Unique variables: {sorted(mnemonics_long['variable'].unique())}")
        
        # Create scenario mnemonics
        print("\nGenerating scenario mnemonics...")
        scenario_mnemonics = []
        scenario_map = [('', 'Baseline'), ('_S1', 'S1'), ('_S3', 'S3'), ('_S4', 'S4')]
        for _, row in mnemonics_long.iterrows():
            baseline = row['ma.ticker']
            if '.' not in baseline:
                print(f"Warning: Baseline mnemonic {baseline} does not contain a dot")
            cinc = row['cinc']
            variable = row['variable']
            for suffix, scen in scenario_map:
                new_mnemonic = baseline if suffix == '' else baseline.replace('.','%s.' % suffix)
                scenario_mnemonics.append({
                    'cinc': cinc,
                    'variable': variable,
                    'ma.ticker': new_mnemonic,
                    'scenario': scen
                })
        scenario_df = pd.DataFrame(scenario_mnemonics)
        print(f"Sample mnemonics:\n{scenario_df.head()}")
        
        # Load macro variables
        print("\nLoading macro variables...")
        mv_data = pd.read_csv('Output/2_mv_raw_data.csv')
        
        # Handle all extreme values as missing (robust for all variables)
        extreme_mask = mv_data['value'].abs() > 1e+20
        if extreme_mask.any():
            print(f"Found {extreme_mask.sum()} extreme values (>|1e+20|) in macro data - replacing with NaN")
            mv_data.loc[extreme_mask, 'value'] = np.nan
        
        print(f"Replaced {mv_data['value'].isna().sum()} missing values in macro data")
        # Drop cinc from mv_data since it will come from scenario_df
        mv_data = mv_data.drop('cinc', axis=1)
        # Filter macro data
        mv_data_filtered = mv_data.merge(
            scenario_df[['ma.ticker', 'scenario', 'variable', 'cinc']], 
            on=['ma.ticker', 'scenario'],
            how='inner'
        )
        # Add general macro variable + scenario column
        mv_data_filtered['var_scen'] = mv_data_filtered['variable'] + '_' + mv_data_filtered['scenario']

        # Pivot macro data
        print("\nPivoting macro data to wide format (variable + scenario columns)...")
        mv_data_wide = mv_data_filtered.pivot_table(
            index=['yyyyqq', 'cinc'], 
            columns='var_scen', 
            values='value'
        ).reset_index()
        
        # Convert yyyyqq to datetime
        print("\nConverting dates to datetime format...")
        mv_data_wide['yyyyqq'] = pd.to_datetime(mv_data_wide['yyyyqq'], errors='coerce')
        
        # Merge PD and macro data
        print("\nMerging PD and macro data...")
        regression_data = pd.merge(
            mv_data_wide,
            sov_pd_q[['yyyyqq', 'cinc', 'pid', 'entityName', 'region', 'cdsiedf5', 'cdsiedf5_final', 'cdsiedf5_sa', 'cdsiedf5_seasonal', 'cdsiedf5_sa_method']],
            on=['yyyyqq', 'cinc'],
            how='left'
        )
        
        # Sort and calculate transformations using seasonally adjusted PD
        regression_data = regression_data.sort_values(['cinc', 'yyyyqq'])
        print("\nCalculating lnPD and dlnPD transformations using seasonally adjusted PD...")
        
        # Use seasonally adjusted PD for log transformations
        regression_data['lnPD'] = np.log(regression_data['cdsiedf5_final'])
        regression_data['lnPD_lag'] = regression_data.groupby('cinc')['lnPD'].shift(1)
        regression_data['dlnPD'] = regression_data['lnPD'] - regression_data['lnPD_lag']
        
        # Also calculate transformations for original PD for comparison
        regression_data['lnPD_original'] = np.log(regression_data['cdsiedf5'])
        regression_data['dlnPD_original'] = regression_data.groupby('cinc')['lnPD_original'].diff()
        
        # Calculate mean-reverting term using seasonally adjusted PD
        print("Calculating mean-reverting term...")
        for country in regression_data['cinc'].unique():
            country_mask = regression_data['cinc'] == country
            country_data = regression_data[country_mask]
            
            # Calculate TTC (through-the-cycle) PD using historical average of SA PD
            historical_pd = country_data['cdsiedf5_final'].dropna()
            if len(historical_pd) > 0:
                ttc_pd = historical_pd.mean()
                ln_ttc_pd = np.log(ttc_pd)
                
                # Mean-reverting term: ln(TTC_PD) - lnPD_lag
                regression_data.loc[country_mask, 'mean_reverting'] = ln_ttc_pd - regression_data.loc[country_mask, 'lnPD_lag']
        
        # Final summary
        print("\nFinal dataset summary:")
        print(f"Time period: {regression_data['yyyyqq'].min()} to {regression_data['yyyyqq'].max()}")
        print(f"Number of countries: {regression_data['cinc'].nunique()}")
        print(f"Total observations: {len(regression_data)}")
        print(f"Observations with original PD data: {regression_data['cdsiedf5'].notna().sum()}")
        print(f"Observations with seasonally adjusted PD: {(regression_data['cdsiedf5_sa_method'] != 'none').sum()}")
        print(f"Missing values in key columns:")
        print(f"  cdsiedf5 (original): {regression_data['cdsiedf5'].isnull().sum()}")
        print(f"  cdsiedf5_final (SA): {regression_data['cdsiedf5_final'].isnull().sum()}")
        print(f"  lnPD: {regression_data['lnPD'].isnull().sum()}")
        print(f"  dlnPD: {regression_data['dlnPD'].isnull().sum()}")
        print(f"  mean_reverting: {regression_data['mean_reverting'].isnull().sum()}")
        
        # Seasonal adjustment summary by country
        sa_summary = regression_data.groupby('cinc')['cdsiedf5_sa_method'].first().value_counts()
        print(f"\nSeasonal adjustment by country:")
        for method, count in sa_summary.items():
            print(f"  {method}: {count} countries")
        
        # Remove rows where all macro variables and PD data are NaN
        macro_cols = [col for col in regression_data.columns if col not in [
            'pid', 'entityName', 'cinc', 'region', 'yyyyqq', 
            'cdsiedf5', 'cdsiedf5_final', 'cdsiedf5_sa', 'cdsiedf5_seasonal', 'cdsiedf5_sa_method',
            'lnPD', 'lnPD_lag', 'dlnPD', 'lnPD_original', 'dlnPD_original', 'mean_reverting'
        ]]
        keep_mask = regression_data[macro_cols + ['cdsiedf5_final']].notna().any(axis=1)
        regression_data = regression_data[keep_mask].reset_index(drop=True)
        print(f"Rows after removing all-NaN macro/PD: {len(regression_data)}")
        
        # Print available macro variables
        print("\nAvailable macro variables:")
        print(macro_cols)
        
        # Save prepared data
        print("\nSaving regression data...")
        regression_data.to_csv('Output/3.regression_data.csv', index=False)
        print("Data preparation complete!")

        # Create PD seasonal adjustment plots
        print("\n" + "="*60)
        print("CREATING PD SEASONAL ADJUSTMENT PLOTS")
        print("="*60)
        output_dir = Path('Output/3.pd_seasonal_adjustment')
        create_pd_seasonal_adjustment_plots(regression_data, output_dir)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 