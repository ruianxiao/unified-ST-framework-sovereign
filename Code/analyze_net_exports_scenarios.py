"""
Net Export Analysis Across Scenarios
=====================================

This script analyzes net export performance across different scenarios (S1, S3, S4),
comparing rank orders and investigating relationships with historical averages.

Key questions addressed:
1. First quarter net export returns by scenario
2. Countries with S1 < S3 < S4 vs reverse ordering
3. Historical averages and their relationship to scenario ordering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path('..')
OUTPUT_DIR = BASE_DIR / 'Output'
TRANSFORMED_DATA_FILE = OUTPUT_DIR / '4.transformation' / 'transformed_data.csv'

# Output directory for this analysis
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / 'net_exports_scenario_analysis'
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)

def load_transformed_data():
    """Load transformed data with scenario forecasts"""
    print("Loading transformed data...")
    
    # Load the data
    data = pd.read_csv(TRANSFORMED_DATA_FILE)
    
    # Convert date column
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    
    print(f"Loaded {len(data)} rows of transformed data")
    print(f"Date range: {data['yyyyqq'].min()} to {data['yyyyqq'].max()}")
    print(f"Countries: {data['cinc'].nunique()}")
    
    return data

def extract_net_exports_data(data):
    """Extract net exports data for all scenarios"""
    print("Extracting net exports data...")
    
    # Net exports columns
    net_exports_cols = [
        'Net Exports_Baseline', 'Net Exports_S1', 'Net Exports_S3', 'Net Exports_S4'
    ]
    
    # Filter for countries that have net exports data
    net_exports_mask = data[net_exports_cols].notna().any(axis=1)
    net_exports_data = data[net_exports_mask].copy()
    
    print(f"Found {len(net_exports_data)} rows with net exports data")
    print(f"Countries with net exports data: {net_exports_data['cinc'].nunique()}")
    
    return net_exports_data

def get_forecast_start_date(data):
    """Determine the forecast start date"""
    # Look for the first date where scenario data differs from baseline
    forecast_start = None
    
    for _, row in data.iterrows():
        if pd.notna(row['Net Exports_Baseline']) and pd.notna(row['Net Exports_S1']):
            if abs(row['Net Exports_Baseline'] - row['Net Exports_S1']) > 1e-10:
                forecast_start = row['yyyyqq']
                break
    
    if forecast_start is None:
        # If no difference found, use a reasonable default
        forecast_start = pd.to_datetime('2025-01-01')
    
    print(f"Forecast start date: {forecast_start}")
    return forecast_start

def calculate_historical_averages(data, forecast_start):
    """Calculate historical averages for net exports before forecast period"""
    print("Calculating historical averages...")
    
    historical_data = data[data['yyyyqq'] < forecast_start].copy()
    
    historical_averages = {}
    
    for country in historical_data['cinc'].unique():
        country_data = historical_data[historical_data['cinc'] == country]
        
        # Use baseline net exports for historical average
        baseline_values = country_data['Net Exports_Baseline'].dropna()
        
        if len(baseline_values) > 0:
            historical_averages[country] = baseline_values.mean()
    
    print(f"Calculated historical averages for {len(historical_averages)} countries")
    return historical_averages

def analyze_first_quarter_scenarios(data, forecast_start):
    """Analyze first quarter of forecast period"""
    print("Analyzing first quarter scenario performance...")
    
    # Get forecast period data
    forecast_data = data[data['yyyyqq'] >= forecast_start].copy()
    
    # Get first quarter of forecast for each country
    first_quarter_data = {}
    
    for country in forecast_data['cinc'].unique():
        country_data = forecast_data[forecast_data['cinc'] == country].sort_values('yyyyqq')
        
        if len(country_data) > 0:
            first_quarter = country_data.iloc[0]
            
            # Extract scenario values
            scenarios = {
                'S1': first_quarter['Net Exports_S1'],
                'S3': first_quarter['Net Exports_S3'],
                'S4': first_quarter['Net Exports_S4'],
                'Baseline': first_quarter['Net Exports_Baseline']
            }
            
            # Only include if we have all scenario data
            if all(pd.notna(val) for val in scenarios.values()):
                first_quarter_data[country] = scenarios
    
    print(f"First quarter scenario data available for {len(first_quarter_data)} countries")
    return first_quarter_data

def rank_countries_by_scenario(first_quarter_data, scenarios=['S1', 'S3', 'S4']):
    """Rank countries by their net export performance in each scenario"""
    print("Ranking countries by scenario performance...")
    
    rankings = {}
    
    for scenario in scenarios:
        # Extract values for this scenario
        scenario_values = {
            country: data[scenario] 
            for country, data in first_quarter_data.items() 
            if scenario in data and pd.notna(data[scenario])
        }
        
        # Sort by value (descending - higher net exports is better)
        sorted_countries = sorted(scenario_values.items(), key=lambda x: x[1], reverse=True)
        
        # Create ranking (1 = best/highest net exports, higher = worse)
        rankings[scenario] = {
            country: rank + 1 
            for rank, (country, value) in enumerate(sorted_countries)
        }
    
    return rankings

def analyze_scenario_ordering(first_quarter_data, scenarios=['S1', 'S3', 'S4']):
    """Analyze whether countries follow S1 < S3 < S4 or reverse ordering"""
    print("Analyzing scenario ordering patterns...")
    
    ordering_analysis = {
        'increasing': [],  # S1 < S3 < S4
        'decreasing': [],  # S1 > S3 > S4
        'mixed': [],       # Other patterns
    }
    
    for country_code, scenario_values in first_quarter_data.items():
        s1_val = scenario_values['S1']
        s3_val = scenario_values['S3']
        s4_val = scenario_values['S4']
        
        if s1_val < s3_val < s4_val:
            ordering_analysis['increasing'].append(country_code)
        elif s1_val > s3_val > s4_val:
            ordering_analysis['decreasing'].append(country_code)
        else:
            ordering_analysis['mixed'].append(country_code)
    
    return ordering_analysis

def create_ranking_comparison(rankings, first_quarter_data, historical_averages):
    """Create simplified ranking comparison with only essential columns"""
    print("Creating simplified comparison...")
    
    # Create DataFrame for analysis
    comparison_data = []
    
    for country in first_quarter_data.keys():
        # Determine ordering pattern
        s1_val = first_quarter_data[country]['S1']
        baseline_val = first_quarter_data[country]['Baseline']
        s3_val = first_quarter_data[country]['S3']
        s4_val = first_quarter_data[country]['S4']
        
        # Check ordering pattern
        if s1_val < s3_val < s4_val:
            ordering = 'Ascending (S1 < S3 < S4)'
        elif s1_val > s3_val > s4_val:
            ordering = 'Descending (S1 > S3 > S4)'
        else:
            ordering = 'Mixed'
        
        row = {
            'Country': country,
            'S1_Q1_Value': s1_val,
            'Baseline_Q1_Value': baseline_val,
            'S3_Q1_Value': s3_val,
            'S4_Q1_Value': s4_val,
            'Ordering_Pattern': ordering,
            'Historical_Average': historical_averages.get(country, None),
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by country name for easy reading
    comparison_df = comparison_df.sort_values('Country')
    
    return comparison_df

def plot_scenario_analysis(comparison_df, ordering_analysis):
    """Create visualizations for scenario analysis"""
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Net Export Scenario Analysis: First Quarter Performance', fontsize=16)
    
    # Plot 1: Scenario Values Distribution
    ax1 = axes[0, 0]
    scenarios = ['S1_Q1_Value', 'S3_Q1_Value', 'S4_Q1_Value', 'Baseline_Q1_Value']
    colors = ['blue', 'orange', 'red', 'green']
    labels = ['S1', 'S3', 'S4', 'Baseline']
    for i, scenario in enumerate(scenarios):
        values = comparison_df[scenario].dropna()
        ax1.hist(values, alpha=0.7, bins=20, label=labels[i], color=colors[i])
    ax1.set_xlabel('Net Export Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of First Quarter Net Export Values by Scenario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ordering Pattern Distribution
    ax2 = axes[0, 1]
    ordering_counts = comparison_df['Ordering_Pattern'].value_counts()
    colors_pie = ['green', 'red', 'orange']
    ax2.pie(ordering_counts.values, labels=ordering_counts.index, colors=colors_pie, autopct='%1.1f%%')
    ax2.set_title('Scenario Ordering Patterns\n(Net Export Values)')
    
    # Plot 3: Historical Average vs S1 Performance
    ax3 = axes[0, 2]
    valid_hist = comparison_df.dropna(subset=['Historical_Average'])
    if len(valid_hist) > 0:
        ax3.scatter(valid_hist['Historical_Average'], valid_hist['S1_Q1_Value'], alpha=0.7, color='blue')
        ax3.set_xlabel('Historical Average Net Exports')
        ax3.set_ylabel('S1 First Quarter Net Export Value')
        ax3.set_title('Historical Average vs S1 Performance')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scenario Comparison by Ordering Pattern
    ax4 = axes[1, 0]
    for pattern in comparison_df['Ordering_Pattern'].unique():
        pattern_data = comparison_df[comparison_df['Ordering_Pattern'] == pattern]
        if len(pattern_data) > 0:
            ax4.scatter(pattern_data['S1_Q1_Value'], pattern_data['S4_Q1_Value'], 
                       alpha=0.7, label=pattern, s=50)
    ax4.set_xlabel('S1 Net Export Value')
    ax4.set_ylabel('S4 Net Export Value')
    ax4.set_title('S1 vs S4 Performance by Ordering Pattern')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Historical Average by Ordering Pattern
    ax5 = axes[1, 1]
    patterns = comparison_df['Ordering_Pattern'].unique()
    hist_by_pattern = []
    pattern_labels = []
    for pattern in patterns:
        pattern_data = comparison_df[comparison_df['Ordering_Pattern'] == pattern]
        hist_vals = pattern_data['Historical_Average'].dropna()
        if len(hist_vals) > 0:
            hist_by_pattern.append(hist_vals.values)
            pattern_labels.append(pattern)
    
    if hist_by_pattern:
        bp = ax5.boxplot(hist_by_pattern, labels=pattern_labels, patch_artist=True)
        ax5.set_ylabel('Historical Average Net Exports')
        ax5.set_title('Historical Averages by Ordering Pattern')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Country Count by Historical Average Ranges
    ax6 = axes[1, 2]
    valid_hist = comparison_df.dropna(subset=['Historical_Average'])
    if len(valid_hist) > 0:
        # Create bins for historical averages
        bins = [-100, -10, 0, 10, 50, 200]
        valid_hist['Hist_Avg_Bin'] = pd.cut(valid_hist['Historical_Average'], bins=bins)
        bin_counts = valid_hist['Hist_Avg_Bin'].value_counts().sort_index()
        ax6.bar(range(len(bin_counts)), bin_counts.values)
        ax6.set_xticks(range(len(bin_counts)))
        ax6.set_xticklabels([str(x) for x in bin_counts.index], rotation=45)
        ax6.set_ylabel('Number of Countries')
        ax6.set_title('Countries by Historical Average Range')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_DIR / 'net_exports_scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comprehensive analysis plot")

def create_detailed_country_analysis(comparison_df, ordering_analysis, historical_averages):
    """Create detailed country-level analysis"""
    print("Creating detailed country analysis...")
    
    # Countries with S1 < S3 < S4 pattern
    increasing_countries = ordering_analysis['increasing']
    decreasing_countries = ordering_analysis['decreasing']
    mixed_countries = ordering_analysis['mixed']
    
    # Analyze historical averages by pattern
    def get_pattern_stats(countries, pattern_name):
        if countries:
            hist_avgs = [historical_averages.get(c, np.nan) for c in countries]
            hist_avgs = [x for x in hist_avgs if not np.isnan(x)]
            
            if hist_avgs:
                return {
                    'pattern': pattern_name,
                    'count': len(countries),
                    'countries': countries,
                    'hist_avg_mean': np.mean(hist_avgs),
                    'hist_avg_std': np.std(hist_avgs),
                    'hist_avg_min': np.min(hist_avgs),
                    'hist_avg_max': np.max(hist_avgs)
                }
        return {'pattern': pattern_name, 'count': len(countries), 'countries': countries}
    
    pattern_stats = [
        get_pattern_stats(increasing_countries, 'S1 < S3 < S4'),
        get_pattern_stats(decreasing_countries, 'S1 > S3 > S4'),
        get_pattern_stats(mixed_countries, 'Mixed')
    ]
    
    return pattern_stats

def main():
    """Main analysis function"""
    print("="*80)
    print("NET EXPORT SCENARIO ANALYSIS")
    print("="*80)
    
    # Load data
    data = load_transformed_data()
    net_exports_data = extract_net_exports_data(data)
    
    # Get forecast start and historical averages
    forecast_start = get_forecast_start_date(net_exports_data)
    historical_averages = calculate_historical_averages(net_exports_data, forecast_start)
    
    # Analyze scenarios
    first_quarter_data = analyze_first_quarter_scenarios(net_exports_data, forecast_start)
    rankings = rank_countries_by_scenario(first_quarter_data)
    ordering_analysis = analyze_scenario_ordering(first_quarter_data)
    
    # Create simplified analysis
    comparison_df = create_ranking_comparison(rankings, first_quarter_data, historical_averages)
    pattern_stats = create_detailed_country_analysis(comparison_df, ordering_analysis, historical_averages)
    
    # Create visualizations
    plot_scenario_analysis(comparison_df, ordering_analysis)
    
    # Save simplified results
    comparison_df.to_csv(ANALYSIS_OUTPUT_DIR / 'net_exports_simple_summary.csv', index=False)
    
    # Print summary results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nTotal countries analyzed: {len(first_quarter_data)}")
    print(f"Countries with historical net exports data: {len(historical_averages)}")
    
    print("\nScenario Ordering Patterns:")
    pattern_counts = comparison_df['Ordering_Pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} countries")
        
        # Show historical averages for each pattern
        pattern_data = comparison_df[comparison_df['Ordering_Pattern'] == pattern]
        hist_vals = pattern_data['Historical_Average'].dropna()
        if len(hist_vals) > 0:
            print(f"    Historical average: {hist_vals.mean():.3f} ± {hist_vals.std():.3f}")
    
    print("\nTop 5 Countries by Net Export Performance:")
    print("\nS1 Scenario:")
    top_s1 = comparison_df.nlargest(5, 'S1_Q1_Value')[['Country', 'S1_Q1_Value']]
    for _, row in top_s1.iterrows():
        print(f"  {row['Country']}: {row['S1_Q1_Value']:.2f}")
    
    print("\nS4 Scenario:")
    top_s4 = comparison_df.nlargest(5, 'S4_Q1_Value')[['Country', 'S4_Q1_Value']]
    for _, row in top_s4.iterrows():
        print(f"  {row['Country']}: {row['S4_Q1_Value']:.2f}")
    
    print("\nSample of Results (First 10 countries alphabetically):")
    print(comparison_df.head(10)[['Country', 'S1_Q1_Value', 'Baseline_Q1_Value', 'S3_Q1_Value', 'S4_Q1_Value', 'Ordering_Pattern', 'Historical_Average']].to_string(index=False))
    
    print(f"\nOutput files:")
    print(f"- Simple summary: {ANALYSIS_OUTPUT_DIR / 'net_exports_simple_summary.csv'}")
    print(f"- Visualization: {ANALYSIS_OUTPUT_DIR / 'net_exports_scenario_analysis.png'}")
    print(f"- Analysis directory: {ANALYSIS_OUTPUT_DIR}")

if __name__ == "__main__":
    main() 