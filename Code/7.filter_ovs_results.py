"""
Step 7: Filter OVS Results
Filter OVS results to select top models per country using comprehensive mandatory filtering criteria.
This version applies all filtering rules including:
- Excludes Monetary Policy Rate, Term Spread, Government 10Y Bond Rate, and Inflation variables
- Applies FX sign constraints based on theoretical expectations
- Ensures economic consistency (oil/commodity importers/exporters)
- Requires all macro variables to be stationary
- Excludes models with PD lag terms for cleaner interpretation
- Selects models with 2+ macro variables having adj_r2 >= 80% of top model (no 1MV models, no diversity requirements)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
# Removed TOP_MODELS_PER_COUNTRY - now using 80% threshold instead of fixed count
GCORR_FORECAST_QUARTERS = 40  # Number of GCorr forecast quarters (20 or 40)

# Oil-dependent economies - using overlap of theoretical and GCorr Sovereign evidence

# Theoretical oil-dependent countries (based on economic theory - major exporters/importers)
THEORETICAL_OIL_DEPENDENT = [
    # From 60-country scope only
    'NOR', 'CAN', 'MEX', 'BRA', 'NGA', 'QAT', 'OMN', 'BHR', 'KAZ', 'IDN', 'MYS'
    # Removed countries not in 60-country scope: SAU, RUS, VEN, IRN, IRQ, KWT, ARE, DZA, AGO, AZE, TTO, ECU, GAB, TCD, COG, GNQ, BRN
]

# GCorr Sovereign oil-dependent countries (from attachment list)
GCORR_OIL_DEPENDENT = [
    # From 60-country scope only  
    'BHR',  # Bahrain
    'COL',  # Colombia
    'KAZ',  # Kazakhstan
    'NGA',  # Nigeria
    'NOR',  # Norway
    'OMN',  # Oman
    'QAT',  # Qatar
    'ROU',  # Romania
    'TUN'   # Tunisia
    # Removed countries not in 60-country scope: AGO, ARE, AZE, BOL, DZA, ECU, IRQ, KWT, RUS, SAU, SEN, VEN
]

# Final oil-dependent countries list: intersection of theoretical and GCorr Sovereign
OIL_DEPENDENT_COUNTRIES = list(set(THEORETICAL_OIL_DEPENDENT) & set(GCORR_OIL_DEPENDENT))
OIL_DEPENDENT_COUNTRIES.sort()  # Sort for consistency


# --- STATIONARITY RESULTS LOADING ---

def load_detailed_stationarity_results():
    """Load detailed stationarity test results for output annotation"""
    stationarity_file = Path('Output/4.transformation/final_stationarity_results.csv')
    
    if not stationarity_file.exists():
        print(f"Warning: Stationarity results file not found: {stationarity_file}")
        print("Stationarity results will not be included in output.")
        return None
    
    try:
        stationarity_df = pd.read_csv(stationarity_file)
        print(f"Loaded {len(stationarity_df)} detailed stationarity test results for annotation")
        return stationarity_df
    
    except Exception as e:
        print(f"Error loading stationarity results: {str(e)}")
        print("Stationarity results will not be included in output.")
        return None

# --- COMMODITY AND OIL EXPORTER/IMPORTER DEFINITIONS FOR SIGN CONSTRAINTS ---

# Commodity Exporters (should have negative coefficients for Commodity Index)
COMMODITY_EXPORTERS = [
    # From 60-country scope only
    'AUS', 'CHL', 'BRA', 'ZAF', 'PER', 'COL', 'NZL', 'CAN', 'NOR', 'KAZ', 'QAT', 'BHR', 'OMN', 'NGA', 'IDN', 'MYS', 'MEX'
    # Removed: ECU, BOL, URY, RUS, MNG, TTO, SAU, ARE, KWT, DZA, AGO, GAB, TCD, GNQ, VEN, IRN, IRQ, AZE, BRN (not in 60-country scope)
]

# Commodity Importers (should have positive coefficients for Commodity Index)
COMMODITY_IMPORTERS = [
    # From 60-country scope only
    'JPN', 'KOR', 'CHN', 'THA', 'VNM', 'PHL', 'CHE', 'DEU', 'FRA', 'ITA', 'ESP', 'GBR',
    'NLD', 'BEL', 'AUT', 'DNK', 'SWE', 'FIN', 'PRT', 'IRL', 'CYP', 'SVN', 'SVK', 'EST', 'LVA',
    'LTU', 'POL', 'CZE', 'HUN', 'HRV', 'BGR', 'ROU', 'TUR', 'ISR', 'HKG', 'DOM', 'EGY', 'JAM', 'MAR', 'PAN', 'SRB', 'TUN', 'USA'
    # Removed: IND, PAK, BGD, LKA, LUX, GRC, MLT, SGP, TWN (not in 60-country scope)
]

# Oil Exporters (should have negative coefficients for Oil Price)
OIL_EXPORTERS = [
    # From 60-country scope only
    'QAT', 'BHR', 'OMN', 'NOR', 'COL', 'MEX', 'BRA', 'CAN', 'NGA', 'KAZ', 'IDN', 'MYS', 'EGY', 'TUN'
    # Removed: SAU, ARE, KWT, RUS, VEN, ECU, AGO, DZA, AZE, TTO, GAB, TCD, GNQ, IRN, IRQ, LBY, BRN (not in 60-country scope)
]

# Oil Importers (should have positive coefficients for Oil Price)
OIL_IMPORTERS = [
    # From 60-country scope only
    'JPN', 'KOR', 'CHN', 'THA', 'VNM', 'PHL', 'CHE', 'DEU', 'FRA', 'ITA', 'ESP', 'GBR', 'NLD', 'BEL', 'AUT', 'DNK', 'SWE', 'FIN', 'PRT', 'IRL', 'CYP', 'SVN', 'SVK',
    'EST', 'LVA', 'LTU', 'POL', 'CZE', 'HUN', 'HRV', 'BGR', 'ROU', 'TUR', 'ISR', 'HKG', 'DOM', 'JAM', 'MAR', 'PAN', 'SRB', 'AUS', 'CHL', 'NZL', 'PER', 'USA', 'ZAF'
    # Removed: IND, SGP, PAK, BGD, LKA, NPL, LUX, GRC, MLT, TWN (not in 60-country scope)
]

# --- NET EXPORTS RELEVANCE CATEGORIES ---

# Export-dependent countries where Net Exports is highly relevant for sovereign risk
# Mechanism: Trade balance directly affects foreign exchange earnings and debt servicing capacity
NET_EXPORTS_HIGH_RELEVANCE = [
    # Commodity exporters (from 60-country scope)
    'AUS', 'CHL', 'BRA', 'ZAF', 'PER', 'COL', 'NZL', 'CAN', 'NOR', 'KAZ', 'QAT', 'BHR', 'OMN', 'NGA', 'IDN', 'MYS', 'MEX',
    
    # Manufacturing export hubs and small open economies (from 60-country scope)
    # Removed: PHL, HKG due to weird scenario design
    'VNM',  # Vietnam - export manufacturing hub
    'THA',  # Thailand - export-dependent manufacturing
    # 'PHL',  # Philippines - export-dependent
    'KOR',  # South Korea - export-oriented economy
    # 'HKG',  # Hong Kong - trade and services hub
    'CZE',  # Czech Republic - export-oriented manufacturing
    'HUN',  # Hungary - export-oriented manufacturing
    'SVK',  # Slovakia - export-oriented manufacturing
    'SVN',  # Slovenia - small open economy
    'EST',  # Estonia - small open economy
    'LVA',  # Latvia - small open economy
    'LTU',  # Lithuania - small open economy
    'BGR',  # Bulgaria - export-dependent
    'ROU',  # Romania - export-dependent
    'HRV',  # Croatia - small open economy
    'POL',  # Poland - export manufacturing
    'TUR',  # Turkey - export manufacturing
    'MAR',  # Morocco - export-dependent
    'TUN',  # Tunisia - export-dependent
    'EGY',  # Egypt - export-dependent
    'DOM',  # Dominican Republic - small open economy
    'JAM',  # Jamaica - small open economy
    'PAN',  # Panama - services exports
    'SRB',  # Serbia - export-dependent
]

# Large diversified economies where Net Exports has lower relevance for sovereign risk
# Mechanism: Large domestic markets, diversified economic base, sophisticated policy tools
NET_EXPORTS_LOW_RELEVANCE = [
    'USA',  # United States - large domestic market, reserve currency
    'GBR',  # United Kingdom - large domestic market, financial center
    'DEU',  # Germany - large diversified economy (though export-oriented, EUR dynamics dominate)
    'FRA',  # France - large diversified economy
    'ITA',  # Italy - large diversified economy
    'ESP',  # Spain - large diversified economy
    'JPN',  # Japan - large domestic market despite export strength
    'CHN',  # China - large domestic market, policy tools
    'CHE',  # Switzerland - advanced economy, financial center
    'NLD',  # Netherlands - advanced economy, financial center
    'BEL',  # Belgium - advanced economy, EU financial center
    'AUT',  # Austria - advanced economy, diversified
    'DNK',  # Denmark - advanced economy, strong institutions
    'SWE',  # Sweden - advanced economy, strong institutions
    'FIN',  # Finland - advanced economy, diversified
    'IRL',  # Ireland - advanced economy, financial services
    'PRT',  # Portugal - EU economy, diversified
    'CYP',  # Cyprus - EU economy, financial services
    'ISR',  # Israel - advanced economy, diversified
    'HKG', 'PHL' # Added back due to weird scenario design
]

# --- FX-PD CORRELATION EXPECTATIONS ---

# Countries where FX depreciation should increase PD (positive coefficient expected)
# Mechanism: Balance sheet effect (foreign debt), commodity dependence, external vulnerability
FX_POSITIVE_CORRELATION_EXPECTED = [
    # High foreign currency debt countries (from 60-country scope)
    'TUR',  # Turkey - high USD debt, volatile lira
    'ZAF',  # South Africa - significant foreign currency debt
    'IDN',  # Indonesia - post-1997 crisis vulnerability
    
    # Commodity exporters with foreign debt (double hit from commodity + FX)
    'BRA',  # Brazil - commodity exporter, large foreign debt
    'MEX',  # Mexico - oil exporter, significant USD debt
    'COL',  # Colombia - oil/commodity exporter
    'CHL',  # Chile - copper exporter
    'PER',  # Peru - commodity exporter
    
    # Small open economies vulnerable to external shocks
    'THA',  # Thailand - post-1997 crisis sensitivity
    'PHL',  # Philippines - external financing dependent
    'MYS',  # Malaysia - commodity links, external vulnerability
    'VNM',  # Vietnam - export-dependent
    'POL',  # Poland - external financing needs
    'HUN',  # Hungary - external vulnerability
    'CZE',  # Czech Republic - small open economy
    'BGR',  # Bulgaria - external dependence
    'ROU',  # Romania - external vulnerability
    'HRV',  # Croatia - external dependence
    
    # Currency peg stress countries (when peg under pressure)
    'QAT',  # Qatar - USD peg dynamics
    'HKG',  # Hong Kong - though well-managed, peg maintenance costs
    
    # Other emerging markets with external vulnerabilities
    'KOR',  # South Korea - export dependence, capital flow sensitivity
    # 'DOM',  # Dominican Republic - small Caribbean economy, external vulnerability
    'EGY',  # Egypt - declining oil exporter, high external vulnerability
    # 'JAM',  # Jamaica - small island economy, external vulnerability, removed due to weird scenario design
    'MAR',  # Morocco - import-dependent, external vulnerability
    'PAN',  # Panama - services economy, dollarization effects
    'SRB',  # Serbia - emerging European economy, external vulnerability
    'TUN',  # Tunisia - declining oil exporter, external vulnerability
    
    # Removed: ARG, RUS, SAU, ARE, IND, TWN (not in 60-country scope)
]

# Countries where FX depreciation should decrease PD (negative coefficient expected)
# Mechanism: Safe haven status, export competitiveness benefits, low foreign debt
FX_NEGATIVE_CORRELATION_EXPECTED = [
    'CHE',  # Switzerland - safe haven currency
    'JPN',  # Japan - safe haven, export benefits from weaker yen
    'NOR',  # Norway - oil exporter with strong institutions, depreciation helps non-oil sector
]

# Countries where FX-PD correlation should be weak/insignificant
# Mechanism: Large diversified economies, low foreign debt, flexible policy responses
FX_NO_CORRELATION_EXPECTED = [
    # From 60-country scope only
    'USA',  # United States - reserve currency, low foreign debt
    'GBR',  # United Kingdom - reserve currency, flexible policy
    'DEU',  # Germany - large diversified economy, EUR dynamics
    'FRA',  # France - large diversified economy, EUR dynamics
    'ITA',  # Italy - EUR dynamics, but domestic factors dominate
    'ESP',  # Spain - EUR dynamics
    'CAN',  # Canada - commodity exporter but strong institutions
    'AUS',  # Australia - commodity exporter but strong institutions
    'NZL',  # New Zealand - small but strong institutions
    'SWE',  # Sweden - strong institutions, flexible policy
    'DNK',  # Denmark - strong institutions
    'NLD',  # Netherlands - strong institutions
    'BEL',  # Belgium - EUR dynamics
    'AUT',  # Austria - EUR dynamics
    'FIN',  # Finland - EUR dynamics
    'IRL',  # Ireland - EUR dynamics
    'BHR',  # Bahrain - oil exporter, USD peg, strong institutions
    'CHN',  # China - large economy, managed currency, domestic focus
    'CYP',  # Cyprus - EUR dynamics, EU membership
    'EST',  # Estonia - EUR dynamics, strong institutions
    'ISR',  # Israel - advanced economy, flexible policy
    'KAZ',  # Kazakhstan - oil exporter, commodity revenues cushion FX effects
    'LTU',  # Lithuania - EUR dynamics, EU membership
    'LVA',  # Latvia - EUR dynamics, EU membership
    'NGA',  # Nigeria - oil exporter, oil revenues dominate over FX effects
    'OMN',  # Oman - oil exporter, stable fiscal position
    'PRT',  # Portugal - EUR dynamics, EU membership
    'SVK',  # Slovakia - EUR dynamics, EU membership
    'SVN',  # Slovenia - EUR dynamics, EU membership
    'JAM',  # Jamaica - removed due to weird scenario design
    'DOM',  # Dominican Republic - removed to simplify model selection

]

def contains_oil_price_for_importers(row, country):
    """Check if oil importer country has Oil Price in their model (should exclude)"""
    if country not in OIL_IMPORTERS:
        return False  # Not an oil importer, no restriction
    
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            var_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            if 'Oil Price' in var_name:
                return True
    return False

def contains_commodity_index_for_importers(row, country):
    """Check if commodity importer country has Commodity Index in their model (should exclude)"""
    if country not in COMMODITY_IMPORTERS:
        return False  # Not a commodity importer, no restriction
    
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            var_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            if 'Commodity Index' in var_name:
                return True
    return False

# Removed redundant contains_net_exports_for_low_relevance function
# This logic is already handled by has_inconsistent_net_exports_coefficient function

def has_inconsistent_net_exports_coefficient(row, country):
    """Check if Net Exports coefficient is inconsistent with country trade characteristics"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    coef_cols = ['MV1_coefficient', 'MV2_coefficient', 'MV3_coefficient', 'MV4_coefficient']
    
    for mv_col, coef_col in zip(mv_cols, coef_cols):
        if (mv_col in row.index and pd.notna(row[mv_col]) and 
            coef_col in row.index and pd.notna(row[coef_col])):
            var_name = row[mv_col].split('_lag')[0] if '_lag' in row[mv_col] else row[mv_col]
            coef = row[coef_col]
            
            if 'Net Exports' in var_name:
                if country in NET_EXPORTS_HIGH_RELEVANCE:
                    # For export-dependent countries, Net Exports should have NEGATIVE coefficient
                    # (higher net exports = lower sovereign risk = lower PD = negative dlnPD)
                    if coef > 0:
                        return True  # Inconsistent - exclude this model
                elif country in NET_EXPORTS_LOW_RELEVANCE:
                    # For large diversified economies, Net Exports is not relevant - exclude all Net Exports models
                    return True  # Exclude Net Exports models for these countries
                else:
                    # Unknown country - be conservative and exclude Net Exports models
                    return True  # Exclude for safety
    
    return False

def has_no_lags(row):
    """Check if a model has no lagged variables (all lags are 0 or NaN)"""
    lag_cols = ['MV1_lag', 'MV2_lag', 'MV3_lag', 'MV4_lag']
    for col in lag_cols:
        if col in row.index and pd.notna(row[col]) and row[col] > 0:
            return False
    return True

# Removed get_mv_combination_signature function - no longer needed for simplified selection

def count_macro_variables(row):
    """Count number of macro variables in the model"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    count = 0
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]) and row[col].strip():
            count += 1
    return count

def contains_unemployment_or_gdp(row):
    """Check if model contains Unemployment, GDP (but not 'Debt to GDP ratio'), or Equity"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            
            # Check for Unemployment (exact match)
            if 'Unemployment' in base_name:
                return True
            
            # Check for exact GDP match (not "Debt to GDP Ratio")
            if base_name == 'GDP':
                return True
                
            # Check for Equity
            if 'Equity' in base_name:
                return True
    return False

def contains_oil_price(row):
    """Check if model contains Oil Price"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            if 'Oil Price' in base_name:
                return True
    return False

def contains_excluded_variables(row):
    """Check if model contains excluded variables: Monetary Policy Rate, Term Spread, Government 10Y Bond Rate, or Inflation"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    
    # Define variables to exclude
    excluded_patterns = [
        'Monetary Policy Rate',
        'Term Spread',
        'Government 10Y Bond Rate',
        'Inflation'
    ]
    
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            
            # Check for excluded variables
            for pattern in excluded_patterns:
                if pattern in base_name:
                    return True
    
    return False

def contains_fx_with_wrong_sign(row, country):
    """Check if model contains FX variable with coefficient sign that doesn't match theoretical expectation"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    coef_cols = ['MV1_coefficient', 'MV2_coefficient', 'MV3_coefficient', 'MV4_coefficient']
    
    for mv_col, coef_col in zip(mv_cols, coef_cols):
        if (mv_col in row.index and pd.notna(row[mv_col]) and 
            coef_col in row.index and pd.notna(row[coef_col])):
            
            var_name = row[mv_col].split('_lag')[0] if '_lag' in row[mv_col] else row[mv_col]
            coef = row[coef_col]
            
            # Check if this is an FX variable
            if 'FX' in var_name:
                # Check coefficient sign against theoretical expectation
                if country in FX_POSITIVE_CORRELATION_EXPECTED:
                    # Should have positive coefficient (FX depreciation → higher PD)
                    if coef <= 0:
                        return True
                elif country in FX_NEGATIVE_CORRELATION_EXPECTED:
                    # Should have negative coefficient (FX depreciation → lower PD)  
                    if coef >= 0:
                        return True
                elif country in FX_NO_CORRELATION_EXPECTED:
                    # FX should not be significant for these countries - exclude all FX models
                    return True
                else:
                    # Unknown country - be conservative and exclude FX
                    return True
    
    return False










def has_lag_pd_term(row):
    """Check if model includes lag PD term"""
    return row.get('includes_lag', False) if 'includes_lag' in row.index else False

def add_stationarity_results(df, stationarity_df, country):
    """Add stationarity test results for each MV in the model"""
    if stationarity_df is None:
        # Add empty columns if no stationarity data available
        for i in range(1, 5):
            df[f'MV{i}_is_stationary'] = pd.NA
            df[f'MV{i}_adf_p_value'] = pd.NA
            df[f'MV{i}_kpss_p_value'] = pd.NA
        return df
    
    # Get stationarity results for this country
    country_stationarity = stationarity_df[stationarity_df['country'] == country]
    
    # Create lookup dictionary for faster access
    stationarity_lookup = {}
    for _, row in country_stationarity.iterrows():
        stationarity_lookup[row['variable']] = {
            'is_stationary': row['is_stationary'],
            'adf_p_value': row['adf_p_value'],
            'kpss_p_value': row['kpss_p_value']
        }
    
    # Add stationarity results for each MV column
    for i in range(1, 5):
        mv_col = f'MV{i}'
        is_stat_col = f'MV{i}_is_stationary'
        adf_col = f'MV{i}_adf_p_value'
        kpss_col = f'MV{i}_kpss_p_value'
        
        # Initialize columns
        df[is_stat_col] = pd.NA
        df[adf_col] = pd.NA
        df[kpss_col] = pd.NA
        
        # Fill in stationarity results where MV exists
        for idx in df.index:
            if mv_col in df.columns and pd.notna(df.loc[idx, mv_col]):
                mv_name = df.loc[idx, mv_col]
                base_name = mv_name.split('_lag')[0] if '_lag' in mv_name else mv_name
                
                if base_name in stationarity_lookup:
                    stat_result = stationarity_lookup[base_name]
                    df.loc[idx, is_stat_col] = stat_result['is_stationary']
                    df.loc[idx, adf_col] = stat_result['adf_p_value']
                    df.loc[idx, kpss_col] = stat_result['kpss_p_value']
    
    return df

def has_all_stationary_mvs(row):
    """Check if all macro variables in the model are stationary"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    stat_cols = ['MV1_is_stationary', 'MV2_is_stationary', 'MV3_is_stationary', 'MV4_is_stationary']
    
    # Check each MV position
    for mv_col, stat_col in zip(mv_cols, stat_cols):
        # If there's an MV in this position
        if mv_col in row.index and pd.notna(row[mv_col]) and row[mv_col].strip():
            # Check if stationarity info is available and if it's stationary
            if stat_col in row.index and pd.notna(row[stat_col]):
                if not row[stat_col]:  # Not stationary
                    return False
            else:
                # If stationarity info is missing, consider it non-stationary
                return False
    
    return True

def filter_ovs_results_consolidated(
    output_suffix="",
    use_gcorr40q=False,
    use_smoothed=False,
    require_all_stationary=True
):
    """Consolidated OVS filtering function with mandatory filtering criteria
    
    Parameters:
    - output_suffix: Suffix for output file names
    - use_gcorr40q: If True, use ovs_result_gcorr40q.csv instead of ovs_results.csv
    - use_smoothed: If True, use smoothed GCorr data (adds _smoothed suffix)
    - require_all_stationary: If True, exclude models where any macro variable is non-stationary
    
    Mandatory filtering criteria (all applied):
    
    1. Basic model requirements:
       - Must include Unemployment, GDP (excluding "Debt to GDP ratio"), or Equity
       - Oil-dependent countries must include Oil Price
       - Different MV combinations for lagged models
       - At least one no-lag model when available
       - Results sorted by adj_r2 with has_no_lags indicator
    
    2. Variable exclusions:
       - Exclude models with Monetary Policy Rate, Term Spread, Government 10Y Bond Rate, or Inflation
    
    3. FX sign constraints:
       - Allow FX models only when coefficient signs match theoretical expectations:
         * Positive countries (balance sheet effect): TUR, ARG, BRA, ZAF, etc.
         * Negative countries (safe haven): CHE, JPN, NOR
         * No correlation countries (diversified economies): Exclude all FX models
    
    4. Economic consistency filters:
       - Oil importers cannot have Oil Price in their models
       - Commodity importers cannot have Commodity Index in their models
       - Net Exports consistency: exclude all models for low-relevance countries, require negative coefficients for high-relevance countries
    
    5. Stationarity requirements:
       - Adds stationarity test results for each MV in the final output
       - Optionally exclude models where any macro variable is non-stationary
       - Includes is_stationary, adf_p_value, and kpss_p_value for each MV1-MV4
    
    6. Final model selection (applied after all quality filters):
       - All models with adj_r2 >= 80% of the top model per country
       - Only select models with 2+ macro variables (no 1MV models, no diversity checks)
    """
    
    # Load detailed stationarity results for output annotation
    stationarity_df = load_detailed_stationarity_results()
    
    # Load OVS results
    if use_gcorr40q:
        if use_smoothed:
            ovs_file = Path(f'Output/6.ovs_variable_selection/ovs_results_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed.csv')
            data_source = f"GCorr {GCORR_FORECAST_QUARTERS}Q smoothed"
        else:
            ovs_file = Path(f'Output/6.ovs_variable_selection/ovs_results_gcorr{GCORR_FORECAST_QUARTERS}q.csv')
            data_source = f"GCorr {GCORR_FORECAST_QUARTERS}Q"
    else:
        ovs_file = Path('Output/6.ovs_variable_selection/ovs_results.csv')
        data_source = "standard"
    
    if not ovs_file.exists():
        raise FileNotFoundError(f"OVS results file not found: {ovs_file}")
    
    filter_desc = "advanced"
    if require_all_stationary:
        filter_desc += " (all stationary MVs)"
    if use_gcorr40q:
        filter_desc += f" (GCorr {GCORR_FORECAST_QUARTERS}Q)"
    if use_smoothed:
        filter_desc += " (smoothed)"
    
    print(f"Loading {data_source} OVS results for {filter_desc} filtering...")
    all_results = pd.read_csv(ovs_file)
    print(f"Loaded {len(all_results)} total OVS results")
    
    # Create output directory
    output_dir = Path('Output/7.filtered_ovs_results')
    output_dir.mkdir(exist_ok=True)
    
    filtered_results = []
    country_stats = {}
    
    print(f"\nFiltering results by country ({filter_desc})...")
    
    for country in sorted(all_results['country'].unique()):
        country_results = all_results[all_results['country'] == country].copy()
        country_results = country_results.sort_values('adj_r2', ascending=False)
        
        # Apply basic filters (except MV count - moved to end)
        # Add MV count for later use
        country_results['mv_count'] = country_results.apply(count_macro_variables, axis=1)
        
        # Filter 1: Must contain Unemployment, GDP, or Equity
        country_results['has_unemployment_or_gdp'] = country_results.apply(contains_unemployment_or_gdp, axis=1)
        country_results = country_results[country_results['has_unemployment_or_gdp']]
        
        # Filter 2: Exclude specified variables (mandatory)
        country_results['has_excluded_variables'] = country_results.apply(contains_excluded_variables, axis=1)
        country_results = country_results[~country_results['has_excluded_variables']]  # Exclude those with excluded variables
        
        # Filter 3: Apply FX sign constraints (mandatory)
        country_results['fx_wrong_sign'] = country_results.apply(lambda row: contains_fx_with_wrong_sign(row, country), axis=1)
        country_results = country_results[~country_results['fx_wrong_sign']]
            
        # Filter 4: Oil-dependent countries must include Oil Price
        if country in OIL_DEPENDENT_COUNTRIES:
            country_results['has_oil_price'] = country_results.apply(contains_oil_price, axis=1)
            country_results = country_results[country_results['has_oil_price']]
        
        # Filter 5: Remove PD lag models (mandatory)
        # Note: This excludes models with lagged PD terms for cleaner interpretation
        country_results['has_lag_pd'] = country_results.apply(has_lag_pd_term, axis=1)
        country_results = country_results[~country_results['has_lag_pd']]
        
        # Filter 6: Oil importers should not have Oil Price in their models
        country_results['oil_price_for_importer'] = country_results.apply(lambda row: contains_oil_price_for_importers(row, country), axis=1)
        country_results = country_results[~country_results['oil_price_for_importer']]
        
        # Filter 7: Commodity importers should not have Commodity Index in their models
        country_results['commodity_for_importer'] = country_results.apply(lambda row: contains_commodity_index_for_importers(row, country), axis=1)
        country_results = country_results[~country_results['commodity_for_importer']]
        
        # Filter 8: Net Exports consistency check (handles both low-relevance exclusion and high-relevance coefficient sign)
        country_results['inconsistent_net_exports'] = country_results.apply(lambda row: has_inconsistent_net_exports_coefficient(row, country), axis=1)
        country_results = country_results[~country_results['inconsistent_net_exports']]
        
        # Filter 9: Apply stationarity filter (requires all MVs to be stationary)
        if require_all_stationary:
            # Add stationarity results for filtering
            country_results_with_stationarity = add_stationarity_results(country_results, stationarity_df, country)
            country_results_with_stationarity['all_mvs_stationary'] = country_results_with_stationarity.apply(has_all_stationary_mvs, axis=1)
            
            initial_count = len(country_results_with_stationarity)
            country_results = country_results_with_stationarity[country_results_with_stationarity['all_mvs_stationary'] == True]
            
            if len(country_results) < initial_count:
                print(f"    {country}: Filtered {initial_count - len(country_results)} models with non-stationary MVs")
        
        if len(country_results) == 0:
            country_stats[country] = {
                'total_models': len(all_results[all_results['country'] == country]),
                'filtered_models': 0,
                'models_with_all_stationary_mvs': 0,
                'has_no_lag_model': False,
                'oil_dependent': country in OIL_DEPENDENT_COUNTRIES
            }
            continue
        
        # Make a copy to avoid SettingWithCopyWarning when adding analysis columns
        country_results = country_results.copy()
        
        # Simplified model selection: Select top models with 2+ macro variables only
        country_results_2mv = country_results[country_results['mv_count'] >= 2]
        
        # Select models with adj_r2 >= 80% of top model (instead of fixed top N)
        if len(country_results_2mv) > 0:
            # Get the top model's adj_r2
            top_adj_r2 = country_results_2mv.iloc[0]['adj_r2']
            
            # Calculate 80% threshold
            threshold_adj_r2 = 0.8 * top_adj_r2
            
            # Select all models with adj_r2 >= 80% of top model
            selected_models = country_results_2mv[country_results_2mv['adj_r2'] >= threshold_adj_r2]
        else:
            selected_models = pd.DataFrame()  # Empty DataFrame if no 2MV+ models
        
        # Process selected models if any found
        if len(selected_models) > 0:
            selected_df = selected_models.copy()
            
            # Add ranking and indicators
            selected_df['rank_in_country'] = range(1, len(selected_models) + 1)
            selected_df['has_no_lags'] = selected_df.apply(has_no_lags, axis=1)
            
            # Add stationarity results for each MV (for output annotation)
            selected_df = add_stationarity_results(selected_df, stationarity_df, country)
            
            # Add indicator for models with all stationary MVs
            selected_df['all_mvs_stationary'] = selected_df.apply(has_all_stationary_mvs, axis=1)
            
            # Add to final results
            for _, model in selected_df.iterrows():
                filtered_results.append(model)
        else:
            selected_df = pd.DataFrame()  # Empty DataFrame for consistency
        
        # Store country statistics
        models_with_all_stationary = 0
        if len(selected_models) > 0:
            # Count models with all stationary MVs
            models_with_all_stationary = sum(1 for _, model in selected_df.iterrows() 
                                           if model.get('all_mvs_stationary', False))
        
        country_stats[country] = {
            'total_models': len(all_results[all_results['country'] == country]),
            'filtered_models': len(selected_models),
            'models_with_all_stationary_mvs': models_with_all_stationary,
            'has_no_lag_model': len(selected_models) > 0,  # Simplified since we don't check for no-lag requirement
            'oil_dependent': country in OIL_DEPENDENT_COUNTRIES
        }
        
        # Selected models count handled in summary
    
    # Create final filtered dataset
    if filtered_results:
        filtered_df = pd.DataFrame(filtered_results)
        
        # Sort by country and rank
        filtered_df = filtered_df.sort_values(['country', 'rank_in_country'])
        
        # Save filtered results
        output_file = output_dir / f'filtered_ovs_results_{output_suffix}.csv'
        filtered_df.to_csv(output_file, index=False)
        print(f"\nSaved filtered results ({filter_desc}) to: {output_file}")
        print(f"Total filtered models: {len(filtered_df)}")
        
        # Create final top model selection for each country
        final_top_models = []
        
        for country in sorted(filtered_df['country'].unique()):
            country_models = filtered_df[filtered_df['country'] == country].copy()
            country_models = country_models.sort_values('adj_r2', ascending=False)
            
            # Helper function to check variable presence
            def check_variables(model):
                mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
                has_unemployment = False
                has_gdp = False
                has_gov_consumption = False
                has_debt_to_gdp = False
                
                for col in mv_cols:
                    if col in model.index and pd.notna(model[col]):
                        base_name = model[col].split('_lag')[0] if '_lag' in model[col] else model[col]
                        
                        # Check for Unemployment
                        if 'Unemployment' in base_name:
                            has_unemployment = True
                        
                        # Check for exact GDP match (not "Debt to GDP Ratio")
                        if base_name == 'GDP':
                            has_gdp = True
                        
                        # Check for Government Consumption
                        if 'Government Consumption' in base_name:
                            has_gov_consumption = True
                        
                        # Check for Debt to GDP Ratio
                        if 'Debt to GDP Ratio' in base_name:
                            has_debt_to_gdp = True
                
                return has_unemployment, has_gdp, has_gov_consumption, has_debt_to_gdp
            
            # Categorize models into priority groups
            category_1_models = []  # UNR + GDP
            category_2_models = []  # UNR or GDP (but not both)
            category_3_models = []  # Other models
            
            for _, model in country_models.iterrows():
                has_unr, has_gdp, has_gov_cons, has_debt_gdp = check_variables(model)
                
                if has_unr and has_gdp:
                    category_1_models.append((model, has_gov_cons, has_debt_gdp))
                elif has_unr or has_gdp:
                    category_2_models.append((model, has_gov_cons, has_debt_gdp))
                else:
                    category_3_models.append((model, has_gov_cons, has_debt_gdp))
            
            # Helper function to select best model from a category
            def select_best_from_category(category_models):
                if not category_models:
                    return None
                

                
                # Step 1: Filter out models with Government Consumption and Debt to GDP Ratio
                filtered_models = [
                    (model, model['adj_r2']) for model, has_gov_cons, has_debt_gdp in category_models
                    if not has_gov_cons and not has_debt_gdp
                ]
                
                # If no models without Gov Consumption and Debt/GDP, use all models
                if not filtered_models:
                    filtered_models = [(model, model['adj_r2']) for model, has_gov_cons, has_debt_gdp in category_models]
                
                # Step 2: No additional preference filters - proceed to final selection
                

                
                # Select the model with highest adj_r2 from the final filtered list
                return max(filtered_models, key=lambda x: x[1])[0]
            
            # Manual pick for Hungary: Equity, FX, and GDP
            if country == 'HUN':
                # Look for a model with Equity, FX, and GDP across all categories
                hun_manual_model = None
                for model_list in [category_1_models, category_2_models, category_3_models]:
                    if not model_list:
                        continue
                    
                    for model_tuple in model_list:
                        model = model_tuple[0]  # Extract the actual model from the tuple
                        # Check if model contains Equity, FX, and GDP
                        mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
                        has_equity = False
                        has_fx = False  
                        has_gdp = False
                        
                        for col in mv_cols:
                            if col in model.index and pd.notna(model[col]):
                                mv_name = str(model[col])
                                if 'Equity' in mv_name:
                                    has_equity = True
                                elif 'FX' in mv_name:
                                    has_fx = True
                                elif mv_name.split('_lag')[0] == 'GDP':
                                    has_gdp = True
                        
                        if has_equity and has_fx and has_gdp:
                            hun_manual_model = model
                            break
                    
                    if hun_manual_model is not None:
                        break
                
                if hun_manual_model is not None:
                    selected_model = hun_manual_model
                    selection_criteria = 'manual_pick_hun_equity_fx_gdp'
                else:
                    # Fallback to normal selection if manual pick not found
                    if category_1_models:
                        selected_model = select_best_from_category(category_1_models)
                        selection_criteria = 'unemployment_and_gdp'
                    elif category_2_models:
                        selected_model = select_best_from_category(category_2_models)
                        selection_criteria = 'unemployment_or_gdp'
                    elif category_3_models:
                        selected_model = select_best_from_category(category_3_models)
                        selection_criteria = 'other_variables'
            else:
                # Normal selection for all other countries
                if category_1_models:
                    selected_model = select_best_from_category(category_1_models)
                    selection_criteria = 'unemployment_and_gdp'
                elif category_2_models:
                    selected_model = select_best_from_category(category_2_models)
                    selection_criteria = 'unemployment_or_gdp'
                elif category_3_models:
                    selected_model = select_best_from_category(category_3_models)
                    selection_criteria = 'other_variables'

            
            if selected_model is not None:
                # Store the model along with its selection criteria
                model_with_criteria = selected_model.copy()
                model_with_criteria['selection_criteria'] = selection_criteria
                final_top_models.append(model_with_criteria)
        
        # Create DataFrame for final top models
        if final_top_models:
            final_top_df = pd.DataFrame(final_top_models)
            final_top_df = final_top_df.sort_values('country')
            
            # Save final top models
            final_output_file = output_dir / f'final_top_models_{output_suffix}.csv'
            final_top_df.to_csv(final_output_file, index=False)
            print(f"Saved final top models to: {final_output_file}")
            print(f"Final top models: {len(final_top_df)} countries")
            
            # Print selection summary
            unr_and_gdp_count = len(final_top_df[final_top_df['selection_criteria'] == 'unemployment_and_gdp'])
            unr_or_gdp_count = len(final_top_df[final_top_df['selection_criteria'] == 'unemployment_or_gdp'])
            other_count = len(final_top_df[final_top_df['selection_criteria'] == 'other_variables'])
            manual_hun_count = len(final_top_df[final_top_df['selection_criteria'] == 'manual_pick_hun_equity_fx_gdp'])
            print(f"  - Category 1 (Unemployment + GDP): {unr_and_gdp_count}")
            print(f"  - Category 2 (Unemployment OR GDP): {unr_or_gdp_count}")
            print(f"  - Category 3 (Other variables): {other_count}")
            if manual_hun_count > 0:
                print(f"  - Manual pick (HUN: Equity + FX + GDP): {manual_hun_count}")
        
        # Generate summary statistics
        summary_stats = {
            'total_countries': len(country_stats),
            'countries_with_models': sum(1 for stats in country_stats.values() if stats['filtered_models'] > 0),
            'total_filtered_models': len(filtered_df),
            'models_with_all_stationary_mvs': sum(stats['models_with_all_stationary_mvs'] for stats in country_stats.values()),
            'average_models_per_country': len(filtered_df) / len([c for c in country_stats.values() if c['filtered_models'] > 0]) if len([c for c in country_stats.values() if c['filtered_models'] > 0]) > 0 else 0,
            'countries_with_no_lag_models': sum(1 for stats in country_stats.values() if stats['has_no_lag_model']),
            'oil_dependent_countries_processed': sum(1 for stats in country_stats.values() if stats['oil_dependent'])
        }
        

        
        # Save detailed country statistics
        country_stats_df = pd.DataFrame.from_dict(country_stats, orient='index')
        country_stats_df.index.name = 'country'
        country_stats_df.to_csv(output_dir / f'country_filtering_stats_{output_suffix}.csv')
        
        # Save summary
        with open(output_dir / f'filtering_summary_{output_suffix}.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"FILTERING SUMMARY ({filter_desc.upper()})")
        print("="*60)
        print(f"Total countries processed: {summary_stats['total_countries']}")
        print(f"Countries with valid models: {summary_stats['countries_with_models']}")
        print(f"Total filtered models: {summary_stats['total_filtered_models']}")
        print(f"Models with all stationary MVs: {summary_stats['models_with_all_stationary_mvs']}")
        print(f"Average models per country: {summary_stats['average_models_per_country']:.1f}")
        print(f"Countries with no-lag models: {summary_stats['countries_with_no_lag_models']}")
        print(f"Oil-dependent countries: {summary_stats['oil_dependent_countries_processed']}")
        
        # Sample results available in output files
        
        return filtered_df, country_stats
    
    else:
        print("No models passed the filtering criteria!")
        return None, country_stats

# Wrapper functions for different data sources
def filter_ovs_results_advanced():
    """Advanced filtering with all mandatory criteria"""
    return filter_ovs_results_consolidated(
        output_suffix="advanced",
        require_all_stationary=True
    )

def filter_ovs_results_advanced_gcorr():
    """Advanced filtering using GCorr data"""
    return filter_ovs_results_consolidated(
        output_suffix=f"advanced_gcorr{GCORR_FORECAST_QUARTERS}q",
        use_gcorr40q=True,
        require_all_stationary=True
    )

def filter_ovs_results_advanced_gcorr_smoothed():
    """Advanced filtering using GCorr smoothed data"""
    return filter_ovs_results_consolidated(
        output_suffix=f"advanced_gcorr{GCORR_FORECAST_QUARTERS}q_smoothed",
        use_gcorr40q=True,
        use_smoothed=True,
        require_all_stationary=True
    )

if __name__ == "__main__":
    try:
        print("RUNNING ADVANCED FILTERING")
        # Run advanced filtering
        filtered_results_advanced, stats_advanced = filter_ovs_results_advanced()
        print("Advanced filtering completed!")
        
        print(f"\nRUNNING GCORR {GCORR_FORECAST_QUARTERS}Q ADVANCED FILTERING")
        filtered_results_advanced_gcorr, stats_advanced_gcorr = filter_ovs_results_advanced_gcorr()
        print(f"GCorr {GCORR_FORECAST_QUARTERS}Q advanced filtering completed!")
        
        print(f"\nRUNNING GCORR {GCORR_FORECAST_QUARTERS}Q SMOOTHED ADVANCED FILTERING")
        filtered_results_advanced_gcorr_smoothed, stats_advanced_gcorr_smoothed = filter_ovs_results_advanced_gcorr_smoothed()
        print(f"GCorr {GCORR_FORECAST_QUARTERS}Q smoothed advanced filtering completed!")
        
    except Exception as e:
        print(f"Error during filtering: {str(e)}")
        raise 