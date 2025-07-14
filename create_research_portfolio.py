import pandas as pd
import numpy as np
from pathlib import Path

def create_research_portfolio():
    """
    Create a Research Portfolio based on countries from OVS variable selection.
    Each instrument represents one country covered in the sovereign analysis.
    """
    
    # Load OVS results to get list of countries
    print("Loading OVS results...")
    ovs_file = Path('Output/6.ovs_variable_selection/ovs_results.csv')
    
    if not ovs_file.exists():
        print(f"Error: OVS results file not found at {ovs_file}")
        return
    
    # Read OVS results and get unique countries
    ovs_df = pd.read_csv(ovs_file)
    countries = sorted(ovs_df['country'].unique())
    print(f"Found {len(countries)} countries in OVS results:")
    for country in countries:
        print(f"  {country}")
    
    # Load transformed data to get the forecast cutoff date and last PD values
    print("\nLoading transformed data for cutoff PD values...")
    data_file = Path('Output/4.transformation/transformed_data.csv')
    
    if not data_file.exists():
        print(f"Error: Transformed data file not found at {data_file}")
        return
    
    # Read transformed data
    data_df = pd.read_csv(data_file)
    data_df['yyyyqq'] = pd.to_datetime(data_df['yyyyqq'])
    
    # Define forecast cutoff (same as in scenario forecast)
    forecast_start = pd.to_datetime('2025-07-01')  # 2025Q3
    
    # Get the last PD value before forecast cutoff for each country
    historical_data = data_df[data_df['yyyyqq'] < forecast_start]
    
    # Get last available PD for each country
    cutoff_pds = {}
    for country in countries:
        country_data = historical_data[historical_data['cinc'] == country]
        if not country_data.empty:
            # Get last non-null PD value
            pd_data = country_data[country_data['cdsiedf5'].notna()]
            if not pd_data.empty:
                last_pd = pd_data.iloc[-1]['cdsiedf5']
                last_date = pd_data.iloc[-1]['yyyyqq']
                cutoff_pds[country] = {
                    'pd': last_pd,
                    'date': last_date.strftime('%Y-%m-%d')
                }
                print(f"  {country}: PD = {last_pd:.6f} (as of {last_date.strftime('%Y-%m-%d')})")
            else:
                print(f"  {country}: No PD data available")
                cutoff_pds[country] = {'pd': 0.01, 'date': 'No data'}  # Default PD
        else:
            print(f"  {country}: No data available")
            cutoff_pds[country] = {'pd': 0.01, 'date': 'No data'}  # Default PD
    
    # Create portfolio data
    print(f"\nCreating Research Portfolio...")
    
    portfolio_data = []
    
    for i, country in enumerate(countries, 1):
        # Create instrument for Sov MEV sets only
        sov_instrument = {
            'instrumentIdentifier': f'research_{country.lower()}_sov',
            'gcorrMacroeconomicVariableSet': f'{country}_Sov_90',  # Sov MEV format
            'incorporationCountryCode': country,
            'gcorrAssetRSquared': 0.20,  # Standard sovereign RSQ
            'annualizedPDOneYear': cutoff_pds[country]['pd'],
            'amortizedCost': 100.0,  # Standard notional
            'key_setup': 'Sov MEV sets',
            'portfolio_name': 'Research_Portfolio',
            'asset_class': 'Sovereign'
        }
        
        portfolio_data.append(sov_instrument)
    
    # Create DataFrame
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # Save to CSV files
    output_dir = Path('gcorr-research-delivery-validation/Input')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create instrumentReference.csv (portfolio definition)
    instrument_ref_file = output_dir / 'instrumentReference.csv'
    portfolio_df.to_csv(instrument_ref_file, index=False)
    print(f"Saved instrument reference to: {instrument_ref_file}")
    
    # Create instrumentResult.csv (unconditional PD results)
    # This mimics the structure expected by the R script
    result_data = []
    for _, row in portfolio_df.iterrows():
        result_record = {
            'instrumentIdentifier': row['instrumentIdentifier'],
            'scenarioIdentifier': 'Summary',  # Standard identifier for unconditional
            'annualizedPDOneYear': row['annualizedPDOneYear'],
            'amortizedCost': row['amortizedCost']
        }
        result_data.append(result_record)
    
    result_df = pd.DataFrame(result_data)
    instrument_result_file = output_dir / 'instrumentResult.csv'
    result_df.to_csv(instrument_result_file, index=False)
    print(f"Saved instrument results to: {instrument_result_file}")
    
    # Print summary
    print(f"\nResearch Portfolio Summary:")
    print(f"  Total instruments: {len(portfolio_df)}")
    print(f"  Countries covered: {len(countries)}")
    print(f"  Sov MEV instruments: {len(portfolio_df[portfolio_df['key_setup'] == 'Sov MEV sets'])}")
    
    # Show PD statistics
    pd_values = [cutoff_pds[country]['pd'] for country in countries]
    print(f"\nCutoff PD Statistics:")
    print(f"  Min PD: {min(pd_values):.6f}")
    print(f"  Max PD: {max(pd_values):.6f}")
    print(f"  Mean PD: {np.mean(pd_values):.6f}")
    print(f"  Median PD: {np.median(pd_values):.6f}")
    
    return portfolio_df

if __name__ == "__main__":
    portfolio = create_research_portfolio() 