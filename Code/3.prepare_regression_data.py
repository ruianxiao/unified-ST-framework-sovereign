import pandas as pd
import numpy as np
from pandas.tseries.offsets import QuarterEnd
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load sovereign PD data
        logger.info("Loading sovereign PD data...")
        sov_pd = pd.read_csv('Output/1.sov_use_052025.csv')
        logger.info(f"Loaded PD data with shape: {sov_pd.shape}")
        logger.info("PD data columns: %s", sov_pd.columns.tolist())
        
        sov_pd['price_date'] = pd.to_datetime(sov_pd['price_date'])
        sov_pd['yyyyqq'] = sov_pd['price_date'] + QuarterEnd(0)

        # Convert to quarterly frequency by taking average
        logger.info("\nConverting PD to quarterly frequency...")
        sov_pd_q = sov_pd.groupby(['pid', 'entityName', 'cinc', 'region', 'yyyyqq'])[['cdsiedf5']].mean().reset_index()
        logger.info(f"Quarterly PD data shape: {sov_pd_q.shape}")
        
        # Replace SCG with SRB for Serbia
        logger.info("\nReplacing SCG with SRB for Serbia...")
        sov_pd_q['cinc'] = sov_pd_q['cinc'].replace('SCG', 'SRB')
        logger.info("Updated country codes: %s", sorted(sov_pd_q['cinc'].unique()))
        
        # Load macro variables
        logger.info("\nLoading macro variables...")
        mv_data = pd.read_csv('Input/ws_all_macro_2025May28.csv')
        # Replace -3.4E+38 and similar with NaN
        mv_data['value'] = mv_data['value'].replace([-3.4e+38, -3.4028235e+38, -3.40282346638529e+38], np.nan)
        logger.info(f"Loaded macro data with shape: {mv_data.shape}")
        logger.info("Macro data columns: %s", mv_data.columns.tolist())

        # Load mnemonics mapping
        mv_mnemonics = pd.read_csv('Output/2.sov_mv_mnemonics.csv')
        logger.info(f"\nLoaded mnemonics mapping with shape: {mv_mnemonics.shape}")
        logger.info("Mnemonics columns: %s", mv_mnemonics.columns.tolist())

        # Get list of all required mnemonics
        mv_cols = []
        for col in mv_mnemonics.columns:
            if col not in ['cinc', 'entityName']:
                mv_cols.extend(mv_mnemonics[col].dropna().unique())
        mv_cols = list(set(mv_cols))
        logger.info(f"\nNumber of unique mnemonics to extract: {len(mv_cols)}")
        logger.info("Sample mnemonics: %s", mv_cols[:5])

        # Filter for baseline scenario (S1)
        mv_data = mv_data[mv_data['scenario'] == '_S1']
        
        # Remove rows with NaN in time column
        mv_data = mv_data.dropna(subset=['time'])

        # Convert time to datetime
        def convert_to_date_string(x):
            year = int(float(x))
            decimal = float(x) % 1
            # Map decimal to quarter (1-4)
            # 0.00 -> Q1, 0.25 -> Q2, 0.50 -> Q3, 0.75 -> Q4
            quarter = int(decimal * 4) + 1
            # Convert quarter to month (Q1->3, Q2->6, Q3->9, Q4->12)
            month = quarter * 3
            return f"{year}-{month:02d}-01"

        mv_data['yyyyqq'] = pd.to_datetime(mv_data['time'].astype(str).apply(lambda x: f"{float(x):.2f}").apply(convert_to_date_string))
        mv_data['yyyyqq'] = mv_data['yyyyqq'] + QuarterEnd(0)

        # Filter macro data to only required mnemonics
        mv_data['mnemonic_base'] = mv_data['mnemonic'].str.split('_').str[0]
        mv_data = mv_data[mv_data['mnemonic_base'].isin([m.split('.')[0] for m in mv_cols])]
        
        # Create country codes from region column (assuming region is in the format IXXX)
        mv_data['cinc'] = mv_data['region'].str[1:]
        
        # Handle duplicates by taking mean before pivot
        logger.info("\nPreparing macro variables...")
        logger.info("Unique values in mnemonic column: %d", mv_data['mnemonic'].nunique())
        
        # Check unique countries in both datasets
        logger.info("\nUnique countries in PD data: %d", sov_pd_q['cinc'].nunique())
        logger.info("Sample countries in PD data: %s", sov_pd_q['cinc'].unique()[:5])
        logger.info("\nUnique countries in macro data: %d", mv_data['cinc'].nunique())
        logger.info("Sample countries in macro data: %s", mv_data['cinc'].unique()[:5])
        
        # Handle duplicates by taking mean before pivot
        mv_data_agg = mv_data.groupby(['yyyyqq', 'cinc', 'mnemonic_base'])['value'].mean().reset_index()
        
        # Pivot macro variables to wide format
        mv_data_wide = mv_data_agg.pivot(index=['yyyyqq', 'cinc'], columns='mnemonic_base', values='value').reset_index()
        logger.info(f"\nWide macro data shape: {mv_data_wide.shape}")

        # Create a complete time series for each country
        logger.info("\nCreating complete time series...")
        # Get the full date range
        date_range = pd.date_range(
            start=mv_data_wide['yyyyqq'].min(),
            end=mv_data_wide['yyyyqq'].max(),
            freq='Q'
        )
        
        # Create a complete index for all country-date combinations
        countries = mv_data_wide['cinc'].unique()
        complete_index = pd.MultiIndex.from_product(
            [date_range, countries],
            names=['yyyyqq', 'cinc']
        )
        
        # Reindex macro data to include all dates
        mv_data_complete = mv_data_wide.set_index(['yyyyqq', 'cinc']).reindex(complete_index).reset_index()
        
        # Merge PD and macro data, keeping all macro history
        logger.info("\nMerging PD and macro data...")
        regression_data = pd.merge(
            mv_data_complete,
            sov_pd_q[['yyyyqq', 'cinc', 'pid', 'entityName', 'region', 'cdsiedf5']],
            on=['yyyyqq', 'cinc'],
            how='left'
        )
        
        # Load oil price data
        logger.info("\nLoading oil price data...")
        oil_data = pd.read_csv('Input/ws_all_macro_2025May28.csv')
        oil_data = oil_data[
            (oil_data['mnemonic'] == 'FCPWTI_S1.IUSA') & 
            (oil_data['scenario'] == '_S1')
        ].copy()
        
        # Convert time to datetime
        oil_data['yyyyqq'] = pd.to_datetime(oil_data['time'].astype(str).apply(lambda x: f"{float(x):.2f}").apply(convert_to_date_string))
        oil_data['yyyyqq'] = oil_data['yyyyqq'] + QuarterEnd(0)
        
        # Aggregate to quarterly frequency by taking mean
        oil_data = oil_data.groupby('yyyyqq')['value'].mean().reset_index()
        oil_data.rename(columns={'value': 'FCPWTI.IUSA'}, inplace=True)
        
        # Verify oil price data
        logger.info(f"Loaded oil price data with shape: {oil_data.shape}")
        logger.info("Oil price date range: %s to %s", oil_data['yyyyqq'].min(), oil_data['yyyyqq'].max())
        logger.info("Sample oil prices: %s", oil_data['FCPWTI.IUSA'].head().tolist())
        
        # Merge oil price data (global common variable)
        regression_data = pd.merge(
            regression_data,
            oil_data,
            on='yyyyqq',
            how='left'
        )
        
        # Sort by country and date
        regression_data = regression_data.sort_values(['cinc', 'yyyyqq'])
        
        # Calculate lnPD and dlnPD after all data is merged
        logger.info("\nCalculating lnPD and dlnPD transformations...")
        regression_data['lnPD'] = np.log(regression_data['cdsiedf5'])
        regression_data['lnPD_lag'] = regression_data.groupby('cinc')['lnPD'].shift(1)
        regression_data['dlnPD'] = regression_data['lnPD'] - regression_data['lnPD_lag']
        
        logger.info(f"Final regression data shape: {regression_data.shape}")

        # Save prepared data
        logger.info("\nSaving regression data...")
        regression_data.to_csv('Output/3.regression_data.csv', index=False)

        logger.info("Data preparation complete!")

        # Print summary statistics
        logger.info("\nDataset summary:")
        logger.info(f"Time period: {regression_data['yyyyqq'].min()} to {regression_data['yyyyqq'].max()}")
        logger.info(f"Number of countries: {regression_data['cinc'].nunique()}")
        logger.info(f"Total observations: {len(regression_data)}")
        logger.info(f"Observations with PD data: {regression_data['cdsiedf5'].notna().sum()}")
        
        # Print available macro variables
        logger.info("\nAvailable macro variables:")
        macro_cols = [col for col in regression_data.columns if col not in ['pid', 'entityName', 'cinc', 'region', 'yyyyqq', 'cdsiedf5', 'lnPD', 'lnPD_lag', 'dlnPD']]
        logger.info(macro_cols)
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 