import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'Output/4.transformation/transformed_data.csv'
OUTPUT_DIR = Path('Output/mv_convergence_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the macro variable data for convergence analysis"""
    print("Loading macro variable scenario data...")
    data = pd.read_csv(DATA_PATH)
    data['yyyyqq'] = pd.to_datetime(data['yyyyqq'])
    
    # Focus on forecast period starting from 2025-07-01
    forecast_start = pd.to_datetime('2025-07-01')
    forecast_data = data[data['yyyyqq'] >= forecast_start].copy()
    
    print(f"Data loaded: {len(data)} total observations, {len(forecast_data)} forecast observations")
    print(f"Date range: {data['yyyyqq'].min()} to {data['yyyyqq'].max()}")
    
    return data, forecast_data

def identify_mv_scenarios():
    """Identify macro variable columns with scenario patterns"""
    data, _ = load_and_prepare_data()
    
    # Find all macro variable columns with scenario patterns
    mv_scenarios = {}
    scenarios = ['Baseline', 'S1', 'S3', 'S4']
    
    for col in data.columns:
        for scenario in scenarios:
            if col.endswith(f'_{scenario}'):
                base_name = col.replace(f'_{scenario}', '')
                if base_name not in mv_scenarios:
                    mv_scenarios[base_name] = {}
                mv_scenarios[base_name][scenario] = col
    
    # Filter to complete sets (have all 4 scenarios)
    complete_mvs = {mv: scenarios_dict for mv, scenarios_dict in mv_scenarios.items() 
                   if len(scenarios_dict) == 4}
    
    print(f"Found {len(complete_mvs)} macro variables with complete scenario sets:")
    for mv in sorted(complete_mvs.keys()):
        print(f"  {mv}")
    
    return complete_mvs

def calculate_scenario_differences(data, mv_name, scenario_cols):
    """Calculate differences between scenarios for a macro variable"""
    scenarios = ['Baseline', 'S1', 'S3', 'S4']
    differences = {}
    
    # Calculate all pairwise differences
    for i, scen1 in enumerate(scenarios):
        for j, scen2 in enumerate(scenarios):
            if i < j:  # Avoid duplicates
                col1 = scenario_cols[scen1]
                col2 = scenario_cols[scen2]
                
                if col1 in data.columns and col2 in data.columns:
                    diff_name = f'{scen1}_vs_{scen2}'
                    differences[diff_name] = abs(data[col1] - data[col2])
    
    return differences

def analyze_convergence_by_variable(data, mv_scenarios):
    """Analyze convergence patterns for each macro variable"""
    convergence_results = []
    
    print("\nAnalyzing convergence by macro variable...")
    
    for mv_name, scenario_cols in mv_scenarios.items():
        print(f"\nProcessing {mv_name}...")
        
        # Calculate scenario differences
        differences = calculate_scenario_differences(data, mv_name, scenario_cols)
        
        if not differences:
            continue
        
        # Analyze convergence for each country
        for country in data['cinc'].unique():
            country_data = data[data['cinc'] == country].copy()
            
            if len(country_data) < 10:  # Need sufficient data
                continue
            
            country_data = country_data.sort_values('yyyyqq')
            
            # Calculate differences for this country
            country_differences = {}
            for diff_name, diff_series in differences.items():
                country_diff = diff_series[country_data.index]
                country_differences[diff_name] = country_diff
            
            # Find convergence points (when differences become small)
            convergence_thresholds = [0.1, 0.05, 0.01]  # Different convergence criteria
            
            for threshold in convergence_thresholds:
                for diff_name, diff_series in country_differences.items():
                    # Remove NaN values
                    valid_data = diff_series.dropna()
                    if len(valid_data) < 5:
                        continue
                    
                    # Find first point where difference stays below threshold for 4+ quarters
                    convergence_quarter = None
                    for i in range(len(valid_data) - 3):
                        if all(valid_data.iloc[i:i+4] <= threshold):
                            convergence_quarter = i + 1  # 1-indexed
                            break
                    
                    convergence_results.append({
                        'macro_variable': mv_name,
                        'country': country,
                        'scenario_comparison': diff_name,
                        'threshold': threshold,
                        'convergence_quarter': convergence_quarter,
                        'max_difference': valid_data.max() if len(valid_data) > 0 else np.nan,
                        'final_difference': valid_data.iloc[-1] if len(valid_data) > 0 else np.nan,
                        'data_points': len(valid_data)
                    })
    
    return pd.DataFrame(convergence_results)

def analyze_overall_convergence_patterns(convergence_df):
    """Analyze overall convergence patterns across all variables and countries"""
    print("\n" + "="*60)
    print("OVERALL CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Filter to valid convergence results
    valid_convergence = convergence_df[convergence_df['convergence_quarter'].notna()]
    
    if len(valid_convergence) == 0:
        print("No convergence patterns found with current criteria")
        return
    
    # Overall statistics by threshold
    print("\nCONVERGENCE BY THRESHOLD:")
    for threshold in [0.1, 0.05, 0.01]:
        threshold_data = valid_convergence[valid_convergence['threshold'] == threshold]
        if len(threshold_data) > 0:
            median_quarters = threshold_data['convergence_quarter'].median()
            mean_quarters = threshold_data['convergence_quarter'].mean()
            q25 = threshold_data['convergence_quarter'].quantile(0.25)
            q75 = threshold_data['convergence_quarter'].quantile(0.75)
            
            print(f"\nThreshold {threshold}:")
            print(f"  Cases with convergence: {len(threshold_data)}")
            print(f"  Median convergence: {median_quarters:.1f} quarters")
            print(f"  Mean convergence: {mean_quarters:.1f} quarters")
            print(f"  25th percentile: {q25:.1f} quarters")
            print(f"  75th percentile: {q75:.1f} quarters")
    
    # Convergence by macro variable
    print("\nCONVERGENCE BY MACRO VARIABLE (Threshold 0.05):")
    mv_convergence = valid_convergence[valid_convergence['threshold'] == 0.05].groupby('macro_variable')['convergence_quarter'].agg(['count', 'median', 'mean']).round(1)
    mv_convergence = mv_convergence.sort_values('median')
    
    for mv, stats in mv_convergence.iterrows():
        print(f"  {mv}: {stats['median']:.1f} quarters (median), {stats['count']} cases")
    
    # Convergence by scenario comparison
    print("\nCONVERGENCE BY SCENARIO COMPARISON (Threshold 0.05):")
    scenario_convergence = valid_convergence[valid_convergence['threshold'] == 0.05].groupby('scenario_comparison')['convergence_quarter'].agg(['count', 'median', 'mean']).round(1)
    scenario_convergence = scenario_convergence.sort_values('median')
    
    for comparison, stats in scenario_convergence.iterrows():
        print(f"  {comparison}: {stats['median']:.1f} quarters (median), {stats['count']} cases")
    
    return valid_convergence

def create_convergence_visualizations(convergence_df, forecast_data, mv_scenarios):
    """Create visualizations of convergence patterns"""
    print("\nCreating convergence visualizations...")
    
    # 1. Convergence histogram
    valid_convergence = convergence_df[convergence_df['convergence_quarter'].notna()]
    
    if len(valid_convergence) > 0:
        plt.figure(figsize=(12, 8))
        
        for i, threshold in enumerate([0.1, 0.05, 0.01]):
            plt.subplot(2, 2, i+1)
            threshold_data = valid_convergence[valid_convergence['threshold'] == threshold]
            if len(threshold_data) > 0:
                plt.hist(threshold_data['convergence_quarter'], bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Convergence Distribution (Threshold: {threshold})')
                plt.xlabel('Quarters to Convergence')
                plt.ylabel('Frequency')
                
                median_val = threshold_data['convergence_quarter'].median()
                plt.axvline(median_val, color='red', linestyle='--', 
                           label=f'Median: {median_val:.1f}')
                plt.legend()
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        summary_data = []
        for threshold in [0.1, 0.05, 0.01]:
            threshold_data = valid_convergence[valid_convergence['threshold'] == threshold]
            if len(threshold_data) > 0:
                summary_data.append({
                    'Threshold': threshold,
                    'Median': threshold_data['convergence_quarter'].median(),
                    'Mean': threshold_data['convergence_quarter'].mean(),
                    'Count': len(threshold_data)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            x_pos = range(len(summary_df))
            plt.bar(x_pos, summary_df['Median'], alpha=0.7, label='Median')
            plt.bar([x + 0.3 for x in x_pos], summary_df['Mean'], alpha=0.7, label='Mean', width=0.3)
            plt.xlabel('Convergence Threshold')
            plt.ylabel('Quarters to Convergence')
            plt.title('Summary: Convergence Timeline')
            plt.xticks([x + 0.15 for x in x_pos], summary_df['Threshold'])
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'convergence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Sample time series showing convergence
    plt.figure(figsize=(15, 10))
    
    # Select a few representative variables and countries
    sample_variables = list(mv_scenarios.keys())[:4]  # First 4 variables
    
    for i, mv_name in enumerate(sample_variables):
        plt.subplot(2, 2, i+1)
        
        # Find a country with good data for this variable
        sample_country = None
        for country in forecast_data['cinc'].unique():
            country_data = forecast_data[forecast_data['cinc'] == country]
            scenario_cols = mv_scenarios[mv_name]
            
            # Check if all scenarios have data
            has_data = all(col in country_data.columns and 
                          country_data[col].notna().sum() > 10 
                          for col in scenario_cols.values())
            if has_data:
                sample_country = country
                break
        
        if sample_country:
            country_data = forecast_data[forecast_data['cinc'] == sample_country].sort_values('yyyyqq')
            scenario_cols = mv_scenarios[mv_name]
            
            # Plot scenarios
            for scenario, col in scenario_cols.items():
                if col in country_data.columns:
                    valid_data = country_data[country_data[col].notna()]
                    if len(valid_data) > 0:
                        # Create quarter index starting from 1
                        quarters = range(1, len(valid_data) + 1)
                        plt.plot(quarters, valid_data[col], label=scenario, linewidth=2)
            
            plt.title(f'{mv_name}\n({sample_country})')
            plt.xlabel('Forecast Quarter')
            plt.ylabel('Standardized Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_convergence_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_convergence_report(convergence_df):
    """Generate a detailed convergence report"""
    print("\nGenerating detailed convergence report...")
    
    # Save full results
    convergence_df.to_csv(OUTPUT_DIR / 'mv_convergence_analysis.csv', index=False)
    
    # Create summary statistics
    valid_convergence = convergence_df[convergence_df['convergence_quarter'].notna()]
    
    # Overall summary
    summary_stats = {}
    for threshold in [0.1, 0.05, 0.01]:
        threshold_data = valid_convergence[valid_convergence['threshold'] == threshold]
        if len(threshold_data) > 0:
            summary_stats[f'threshold_{threshold}'] = {
                'count': len(threshold_data),
                'median_quarters': threshold_data['convergence_quarter'].median(),
                'mean_quarters': threshold_data['convergence_quarter'].mean(),
                'min_quarters': threshold_data['convergence_quarter'].min(),
                'max_quarters': threshold_data['convergence_quarter'].max(),
                'q25': threshold_data['convergence_quarter'].quantile(0.25),
                'q75': threshold_data['convergence_quarter'].quantile(0.75)
            }
    
    # Variable-specific summary
    variable_summary = {}
    for mv in valid_convergence['macro_variable'].unique():
        mv_data = valid_convergence[
            (valid_convergence['macro_variable'] == mv) & 
            (valid_convergence['threshold'] == 0.05)
        ]
        if len(mv_data) > 0:
            variable_summary[mv] = {
                'count': len(mv_data),
                'median_quarters': mv_data['convergence_quarter'].median(),
                'mean_quarters': mv_data['convergence_quarter'].mean()
            }
    
    # Save summary
    import json
    with open(OUTPUT_DIR / 'convergence_summary.json', 'w') as f:
        json.dump({
            'overall_summary': summary_stats,
            'variable_summary': variable_summary,
            'analysis_date': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"- Full analysis: mv_convergence_analysis.csv")
    print(f"- Summary: convergence_summary.json")
    print(f"- Visualizations: convergence_distribution.png, sample_convergence_patterns.png")

def main():
    print("MACRO VARIABLE CONVERGENCE ANALYSIS")
    print("="*50)
    
    # Load data and identify variables
    data, forecast_data = load_and_prepare_data()
    mv_scenarios = identify_mv_scenarios()
    
    if not mv_scenarios:
        print("No macro variables with complete scenario sets found!")
        return
    
    # Analyze convergence
    print(f"\nAnalyzing convergence for {len(mv_scenarios)} macro variables...")
    convergence_df = analyze_convergence_by_variable(forecast_data, mv_scenarios)
    
    if len(convergence_df) == 0:
        print("No convergence analysis results generated!")
        return
    
    # Analyze patterns
    valid_convergence = analyze_overall_convergence_patterns(convergence_df)
    
    # Create visualizations
    create_convergence_visualizations(convergence_df, forecast_data, mv_scenarios)
    
    # Generate report
    generate_convergence_report(convergence_df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main() 