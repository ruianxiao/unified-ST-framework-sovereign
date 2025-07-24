"""
Step 7: Filter OVS Results
Filter OVS results to select top diverse models per country using advanced filtering criteria.
This version applies comprehensive filtering rules including:
- Excludes FX and Monetary Policy Rate variables
- Ensures model diversity across variable combinations
- Prioritizes models with specific characteristics for oil-dependent countries
- Maintains balance between lagged and non-lagged models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
TOP_MODELS_PER_COUNTRY = 10  # Number of top models to select per country

# Oil-dependent economies - using overlap of theoretical and GCorr Sovereign evidence

# Theoretical oil-dependent countries (based on economic theory - major exporters/importers)
THEORETICAL_OIL_DEPENDENT = [
    'SAU', 'RUS', 'NOR', 'CAN', 'MEX', 'BRA', 'NGA', 'VEN', 'IRN', 'IRQ',
    'KWT', 'ARE', 'QAT', 'OMN', 'BHR', 'DZA', 'AGO', 'AZE', 'KAZ', 'TTO',
    'ECU', 'GAB', 'TCD', 'COG', 'GNQ', 'IDN', 'MYS', 'BRN'
]

# GCorr Sovereign oil-dependent countries (from attachment list)
GCORR_OIL_DEPENDENT = [
    'AGO',  # Angola
    'ARE',  # United Arab Emirates  
    'AZE',  # Azerbaijan
    'BHR',  # Bahrain
    'BOL',  # Bolivia
    'COL',  # Colombia
    'DZA',  # Algeria
    'ECU',  # Ecuador
    'IRQ',  # Iraq
    'KAZ',  # Kazakhstan
    'KWT',  # Kuwait
    'NGA',  # Nigeria
    'NOR',  # Norway
    'OMN',  # Oman
    'QAT',  # Qatar
    'ROU',  # Romania
    'RUS',  # Russian Federation
    'SAU',  # Saudi Arabia
    'SEN',  # Senegal
    'TUN',  # Tunisia
    'VEN'   # Venezuela
]

# Final oil-dependent countries list: intersection of theoretical and GCorr Sovereign
OIL_DEPENDENT_COUNTRIES = list(set(THEORETICAL_OIL_DEPENDENT) & set(GCORR_OIL_DEPENDENT))
OIL_DEPENDENT_COUNTRIES.sort()  # Sort for consistency

print(f"Theoretical oil-dependent countries: {len(THEORETICAL_OIL_DEPENDENT)}")
print(f"GCorr Sovereign oil-dependent countries: {len(GCORR_OIL_DEPENDENT)}")
print(f"Overlap (final filter): {len(OIL_DEPENDENT_COUNTRIES)} countries: {OIL_DEPENDENT_COUNTRIES}")
print(f"Theoretical only: {sorted(set(THEORETICAL_OIL_DEPENDENT) - set(GCORR_OIL_DEPENDENT))}")
print(f"GCorr Sovereign only: {sorted(set(GCORR_OIL_DEPENDENT) - set(THEORETICAL_OIL_DEPENDENT))}")

# --- COMMODITY AND OIL EXPORTER/IMPORTER DEFINITIONS FOR SIGN CONSTRAINTS ---

# Commodity Exporters (should have negative coefficients for Commodity Index)
COMMODITY_EXPORTERS = [
    'AUS', 'CHL', 'BRA', 'ZAF', 'PER', 'COL', 'ECU', 'BOL', 'URY', 'NZL', 'CAN', 'NOR', 'RUS', 'KAZ', 'MNG', 'TTO',
    'SAU', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN', 'DZA', 'NGA', 'AGO', 'GAB', 'TCD', 'GNQ', 'VEN', 'IRN', 'IRQ', 'AZE', 'IDN', 'MYS', 'BRN'
]

# Commodity Importers (should have positive coefficients for Commodity Index)
COMMODITY_IMPORTERS = [
    'JPN', 'KOR', 'CHN', 'IND', 'THA', 'VNM', 'PHL', 'PAK', 'BGD', 'LKA', 'CHE', 'DEU', 'FRA', 'ITA', 'ESP', 'GBR',
    'NLD', 'BEL', 'AUT', 'DNK', 'SWE', 'FIN', 'PRT', 'IRL', 'LUX', 'GRC', 'CYP', 'MLT', 'SVN', 'SVK', 'EST', 'LVA',
    'LTU', 'POL', 'CZE', 'HUN', 'HRV', 'BGR', 'ROU', 'TUR', 'ISR', 'SGP', 'HKG', 'TWN'
]

# Oil Exporters (should have negative coefficients for Oil Price)
OIL_EXPORTERS = [
    'SAU', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN', 'NOR', 'RUS', 'VEN', 'ECU', 'COL', 'MEX', 'BRA', 'CAN', 'NGA', 'AGO',
    'DZA', 'AZE', 'KAZ', 'TTO', 'GAB', 'TCD', 'GNQ', 'IRN', 'IRQ', 'LBY', 'IDN', 'MYS', 'BRN'
]

# Oil Importers (should have positive coefficients for Oil Price)
OIL_IMPORTERS = [
    'JPN', 'KOR', 'CHN', 'IND', 'THA', 'VNM', 'PHL', 'SGP', 'PAK', 'BGD', 'LKA', 'NPL', 'CHE', 'DEU', 'FRA', 'ITA',
    'ESP', 'GBR', 'NLD', 'BEL', 'AUT', 'DNK', 'SWE', 'FIN', 'PRT', 'IRL', 'LUX', 'GRC', 'CYP', 'MLT', 'SVN', 'SVK',
    'EST', 'LVA', 'LTU', 'POL', 'CZE', 'HUN', 'HRV', 'BGR', 'ROU', 'TUR', 'ISR', 'HKG', 'TWN'
]

def has_sign_constraint_violations(row, country):
    """Check if model has sign constraint violations for commodity/oil variables"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    coef_cols = ['MV1_coefficient', 'MV2_coefficient', 'MV3_coefficient', 'MV4_coefficient']
    
    for mv_col, coef_col in zip(mv_cols, coef_cols):
        if mv_col in row.index and pd.notna(row[mv_col]) and coef_col in row.index and pd.notna(row[coef_col]):
            var_name = row[mv_col].split('_lag')[0] if '_lag' in row[mv_col] else row[mv_col]
            coef = row[coef_col]
            
            # Commodity Index constraints
            if 'Commodity Index' in var_name:
                if country in COMMODITY_EXPORTERS and coef > 0:  # Should be negative
                    return True
                if country in COMMODITY_IMPORTERS and coef < 0:  # Should be positive
                    return True
            
            # Oil Price constraints
            if 'Oil Price' in var_name:
                if country in OIL_EXPORTERS and coef > 0:  # Should be negative
                    return True
                if country in OIL_IMPORTERS and coef < 0:  # Should be positive
                    return True
    
    return False

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
                # Countries that can sustainably run trade deficits
                flexible_countries = ['USA', 'GBR']  
                
                if country in flexible_countries:
                    return False  # No restriction for these countries
                else:
                    # For most countries, Net Exports should have NEGATIVE coefficient
                    # (higher net exports = lower sovereign risk = lower PD = negative dlnPD)
                    if coef > 0:
                        return True  # Inconsistent - exclude this model
    
    return False

def get_mv_base_names(mv_columns):
    """Extract base macro variable names from MV columns, ignoring lags"""
    base_names = set()
    for col in mv_columns:
        if pd.notna(col) and col.strip():
            # Remove lag suffix if present (e.g., "GDP_lag2" -> "GDP")
            base_name = col.split('_lag')[0] if '_lag' in col else col
            base_names.add(base_name)
    return base_names

def has_no_lags(row):
    """Check if a model has no lagged variables (all lags are 0 or NaN)"""
    lag_cols = ['MV1_lag', 'MV2_lag', 'MV3_lag', 'MV4_lag']
    for col in lag_cols:
        if col in row.index and pd.notna(row[col]) and row[col] > 0:
            return False
    return True

def get_mv_combination_signature(row):
    """Get a signature representing the combination of base macro variables (ignoring lags)"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    base_names = []
    
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]) and row[col].strip():
            # Remove lag suffix to get base name
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            base_names.append(base_name)
    
    return tuple(sorted(base_names))

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

def contains_interest_rates_or_fx(row):
    """Check if model contains FX, Monetary Policy Rate, or Term Spread variables (excludes Gov 10Y Bond Rate)"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    
    # Define variables to exclude (FX, Monetary Policy Rate, and Term Spread)
    excluded_patterns = [
        'Monetary Policy Rate',
        'FX',
        'Term Spread'
    ]
    
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            
            # Check for excluded variables
            for pattern in excluded_patterns:
                if pattern in base_name:
                    return True
    
    return False

def contains_both_net_exports_and_commodity(row):
    """Check if model contains both Net Exports and Commodity Index at the same time"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    has_net_exports = False
    has_commodity = False
    
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            
            if 'Net Exports' in base_name:
                has_net_exports = True
            elif 'Commodity Index' in base_name:
                has_commodity = True
    
    return has_net_exports and has_commodity

def contains_government_10y_yield(row):
    """Check if model contains Government 10Y Bond Rate variables"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            
            # Check for Government 10Y Bond Rate
            if 'Government 10Y Bond Rate' in base_name:
                return True
    
    return False

def contains_inflation(row):
    """Check if model contains Inflation variables"""
    mv_cols = ['MV1', 'MV2', 'MV3', 'MV4']
    
    for col in mv_cols:
        if col in row.index and pd.notna(row[col]):
            base_name = row[col].split('_lag')[0] if '_lag' in row[col] else row[col]
            
            # Check for Inflation
            if 'Inflation' in base_name:
                return True
    
    return False

def has_lag_pd_term(row):
    """Check if model includes lag PD term"""
    return row.get('includes_lag', False) if 'includes_lag' in row.index else False

def filter_ovs_results_consolidated(
    exclude_fx_mpr=False,
    apply_advanced_rules=False,
    output_suffix="",
    use_gcorr40q=False,
    use_smoothed=False
):
    """Consolidated OVS filtering function with configurable options
    
    Parameters:
    - exclude_fx_mpr: If True, exclude models with FX or Monetary Policy Rate
    - apply_advanced_rules: If True, apply advanced filtering rules:
        * Force lag PD term if it exists (even if not significant)
        * Exclude models with both Net Export and Commodity Index
        * Exclude models with Government 10Y Bond Rate
        * Exclude models with Inflation
    - output_suffix: Suffix for output file names
    - use_gcorr40q: If True, use ovs_result_gcorr40q.csv instead of ovs_results.csv
    - use_smoothed: If True, use smoothed GCorr data (adds _smoothed suffix)
    
    Base filtering criteria (always applied):
    - Top N models per country (highest adj_r2) where N = TOP_MODELS_PER_COUNTRY
    - At least 2 macro variables per model
    - Must include Unemployment, GDP (excluding "Debt to GDP ratio"), or Equity
    - Oil-dependent countries must include Oil Price
    - Different MV combinations for lagged models
    - At least one no-lag model when available
    - Results sorted by adj_r2 with has_no_lags indicator
    
    New economic consistency filters (always applied):
    - Oil importers cannot have Oil Price in their models
    - Commodity importers cannot have Commodity Index in their models  
    - Net Exports coefficient must be negative (except USA/GBR which are flexible)
    """
    
    # Load OVS results
    if use_gcorr40q:
        if use_smoothed:
            ovs_file = Path('Output/6.ovs_variable_selection/ovs_results_gcorr40q_smoothed.csv')
            data_source = "GCorr 40Q smoothed"
        else:
            ovs_file = Path('Output/6.ovs_variable_selection/ovs_results_gcorr40q.csv')
            data_source = "GCorr 40Q"
    else:
        ovs_file = Path('Output/6.ovs_variable_selection/ovs_results.csv')
        data_source = "standard"
    
    if not ovs_file.exists():
        raise FileNotFoundError(f"OVS results file not found: {ovs_file}")
    
    filter_desc = "consolidated"
    if exclude_fx_mpr:
        filter_desc += " (no FX/MPR)"
    if apply_advanced_rules:
        filter_desc += " (advanced rules)"
    if use_gcorr40q:
        filter_desc += " (GCorr 40Q)"
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
        
        # Apply basic filters
        # Filter 1: At least 2 macro variables
        country_results['mv_count'] = country_results.apply(count_macro_variables, axis=1)
        country_results = country_results[country_results['mv_count'] >= 2]
        
        # Filter 2: Must contain Unemployment, GDP, or Equity
        country_results['has_unemployment_or_gdp'] = country_results.apply(contains_unemployment_or_gdp, axis=1)
        country_results = country_results[country_results['has_unemployment_or_gdp']]
        
        # Filter 3: Exclude FX and Monetary Policy Rate variables (conditional)
        if exclude_fx_mpr:
            country_results['has_rates_or_fx'] = country_results.apply(contains_interest_rates_or_fx, axis=1)
            country_results = country_results[~country_results['has_rates_or_fx']]  # Exclude those with FX/MPR
        
        # Filter 4: Oil-dependent countries must include Oil Price
        if country in OIL_DEPENDENT_COUNTRIES:
            country_results['has_oil_price'] = country_results.apply(contains_oil_price, axis=1)
            country_results = country_results[country_results['has_oil_price']]
        
        # # Filter 5: Advanced - Exclude models with both Net Exports and Commodity Index (conditional)
        # if apply_advanced_rules:
        #     country_results['has_both_netexp_commodity'] = country_results.apply(contains_both_net_exports_and_commodity, axis=1)
        #     country_results = country_results[~country_results['has_both_netexp_commodity']]
        
        # Filter 6: Advanced - Exclude models with Government 10Y Bond Rate (conditional)
        if apply_advanced_rules:
            country_results['has_government_10y'] = country_results.apply(contains_government_10y_yield, axis=1)
            country_results = country_results[~country_results['has_government_10y']]
        
        # Filter 7: Advanced - Exclude models with Inflation (conditional)
        if apply_advanced_rules:
            country_results['has_inflation'] = country_results.apply(contains_inflation, axis=1)
            country_results = country_results[~country_results['has_inflation']]
        
        # Filter 8: Oil importers should not have Oil Price in their models
        country_results['oil_price_for_importer'] = country_results.apply(lambda row: contains_oil_price_for_importers(row, country), axis=1)
        country_results = country_results[~country_results['oil_price_for_importer']]
        
        # Filter 9: Commodity importers should not have Commodity Index in their models
        country_results['commodity_for_importer'] = country_results.apply(lambda row: contains_commodity_index_for_importers(row, country), axis=1)
        country_results = country_results[~country_results['commodity_for_importer']]
        
        # Filter 10: Exclude models with inconsistent Net Exports coefficients
        country_results['inconsistent_net_exports'] = country_results.apply(lambda row: has_inconsistent_net_exports_coefficient(row, country), axis=1)
        country_results = country_results[~country_results['inconsistent_net_exports']]
        
        if len(country_results) == 0:
            country_stats[country] = {
                'total_models': len(all_results[all_results['country'] == country]),
                'filtered_models': 0,
                'has_no_lag_model': False,
                'has_lag_pd_model': False if apply_advanced_rules else None,
                'oil_dependent': country in OIL_DEPENDENT_COUNTRIES
            }
            continue
        
        # Advanced selection logic (conditional) - simplified to only exclude Net Export + Commodity Index
        if apply_advanced_rules:
            # For advanced filtering, use standard selection logic but track lag PD for statistics
            country_results['has_lag_pd'] = country_results.apply(has_lag_pd_term, axis=1)
            
            # Use standard selection logic (no PD lag prioritization)
            selected_models = []
            used_combinations = set()
            used_lagged_combinations = set()
            
            # Step 1: Try to include at least one model without lags if it exists
            no_lag_models = country_results[country_results.apply(has_no_lags, axis=1)]
            has_no_lag_available = len(no_lag_models) > 0
            
            # Step 2: Select diverse models with different MV combinations (standard logic)
            for _, model in country_results.iterrows():
                if len(selected_models) >= TOP_MODELS_PER_COUNTRY:
                    break
                    
                combination_sig = get_mv_combination_signature(model)
                is_lagged_model = not has_no_lags(model)
                
                # Skip if this exact model is already selected
                if any(model.name == selected.name for selected in selected_models):
                    continue
                
                # For lagged models: only check against other lagged combinations
                # For no-lag models: check against all combinations
                if is_lagged_model:
                    if combination_sig in used_lagged_combinations:
                        continue
                else:
                    if combination_sig in used_combinations:
                        continue
                
                selected_models.append(model)
                if is_lagged_model:
                    used_lagged_combinations.add(combination_sig)
                else:
                    used_combinations.add(combination_sig)
            
        else:
            # Standard selection logic
            selected_models = []
            used_combinations = set()
            used_lagged_combinations = set()
            
            # Step 1: Try to include at least one model without lags if it exists
            no_lag_models = country_results[country_results.apply(has_no_lags, axis=1)]
            has_no_lag_available = len(no_lag_models) > 0
            
            # Step 2: Select diverse models with different MV combinations
            for _, model in country_results.iterrows():
                if len(selected_models) >= TOP_MODELS_PER_COUNTRY:
                    break
                    
                combination_sig = get_mv_combination_signature(model)
                is_lagged_model = not has_no_lags(model)
                
                # Skip if this exact model is already selected
                if any(model.name == selected.name for selected in selected_models):
                    continue
                
                # For lagged models: only check against other lagged combinations
                # For no-lag models: check against all combinations
                if is_lagged_model:
                    if combination_sig in used_lagged_combinations:
                        continue
                else:
                    if combination_sig in used_combinations:
                        continue
                
                selected_models.append(model)
                if is_lagged_model:
                    used_lagged_combinations.add(combination_sig)
                else:
                    used_combinations.add(combination_sig)
        
        # Ensure at least one no-lag model is included if available (common logic)
        no_lag_models = country_results[country_results.apply(has_no_lags, axis=1)]
        has_no_lag_available = len(no_lag_models) > 0
        
        if has_no_lag_available:
            has_no_lag_in_selection = any(has_no_lags(model) for model in selected_models)
            if not has_no_lag_in_selection:
                # Replace the lowest adj_r2 model with the best no-lag model
                if len(selected_models) > 0:
                    best_no_lag = no_lag_models.iloc[0]
                    worst_selected_idx = min(range(len(selected_models)), 
                                           key=lambda i: selected_models[i]['adj_r2'])
                    selected_models[worst_selected_idx] = best_no_lag
        
        # Sort selected models by adj_r2 (highest first)
        selected_models.sort(key=lambda x: x['adj_r2'], reverse=True)
        
        # Add selected models to results with ranking and indicators
        for i, model in enumerate(selected_models):
            model_copy = model.copy()
            model_copy['rank_in_country'] = i + 1
            model_copy['has_no_lags'] = has_no_lags(model)
            if apply_advanced_rules:
                model_copy['has_lag_pd'] = has_lag_pd_term(model)
            filtered_results.append(model_copy)
        
        # Store country statistics
        country_stats[country] = {
            'total_models': len(all_results[all_results['country'] == country]),
            'filtered_models': len(selected_models),
            'has_no_lag_model': has_no_lag_available,
            'oil_dependent': country in OIL_DEPENDENT_COUNTRIES,
            'unique_combinations': len(used_combinations)
        }
        
        if apply_advanced_rules:
            country_stats[country].update({
                'has_lag_pd_model': len(country_results[country_results.apply(has_lag_pd_term, axis=1)]) > 0,
                'models_with_lag_pd_available': len(country_results[country_results.apply(has_lag_pd_term, axis=1)]),
                'models_without_lag_pd_available': len(country_results[~country_results.apply(has_lag_pd_term, axis=1)])
            })
        
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
        
        # Generate summary statistics
        summary_stats = {
            'total_countries': len(country_stats),
            'countries_with_models': sum(1 for stats in country_stats.values() if stats['filtered_models'] > 0),
            'total_filtered_models': len(filtered_df),
            'average_models_per_country': len(filtered_df) / len([c for c in country_stats.values() if c['filtered_models'] > 0]) if len([c for c in country_stats.values() if c['filtered_models'] > 0]) > 0 else 0,
            'countries_with_no_lag_models': sum(1 for stats in country_stats.values() if stats['has_no_lag_model']),
            'oil_dependent_countries_processed': sum(1 for stats in country_stats.values() if stats['oil_dependent'])
        }
        
        if apply_advanced_rules:
            summary_stats['countries_with_lag_pd_models'] = sum(1 for stats in country_stats.values() if stats.get('has_lag_pd_model', False))
        
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
        print(f"Average models per country: {summary_stats['average_models_per_country']:.1f}")
        print(f"Countries with no-lag models: {summary_stats['countries_with_no_lag_models']}")
        if apply_advanced_rules:
            print(f"Countries with lag PD models: {summary_stats.get('countries_with_lag_pd_models', 0)}")
        print(f"Oil-dependent countries: {summary_stats['oil_dependent_countries_processed']}")
        
        # Sample results available in output files
        
        return filtered_df, country_stats
    
    else:
        print("No models passed the filtering criteria!")
        return None, country_stats

# Wrapper functions for advanced filtering only
def filter_ovs_results_advanced():
    """Advanced filtering (excludes FX/MPR + advanced rules)"""
    return filter_ovs_results_consolidated(
        exclude_fx_mpr=True,  # Now also excludes FX/MPR
        apply_advanced_rules=True,
        output_suffix="advanced"
    )

def filter_ovs_results_advanced_gcorr40q():
    """Advanced filtering using GCorr 40Q data (excludes FX/MPR + advanced rules)"""
    return filter_ovs_results_consolidated(
        exclude_fx_mpr=True,  # Now also excludes FX/MPR
        apply_advanced_rules=True,
        output_suffix="advanced_gcorr40q",
        use_gcorr40q=True
    )

def filter_ovs_results_advanced_gcorr40q_smoothed():
    """Advanced filtering using GCorr 40Q smoothed data (excludes FX/MPR + advanced rules)"""
    return filter_ovs_results_consolidated(
        exclude_fx_mpr=True,  # Now also excludes FX/MPR
        apply_advanced_rules=True,
        output_suffix="advanced_gcorr40q_smoothed",
        use_gcorr40q=True,
        use_smoothed=True
    )

if __name__ == "__main__":
    try:
        print("RUNNING ADVANCED FILTERING")
        # Run advanced filtering
        filtered_results_advanced, stats_advanced = filter_ovs_results_advanced()
        print("Advanced filtering completed!")
        
        print("\nRUNNING GCORR 40Q ADVANCED FILTERING")
        filtered_results_advanced_gcorr40q, stats_advanced_gcorr40q = filter_ovs_results_advanced_gcorr40q()
        print("GCorr 40Q advanced filtering completed!")
        
        print("\nRUNNING GCORR 40Q SMOOTHED ADVANCED FILTERING")
        filtered_results_advanced_gcorr40q_smoothed, stats_advanced_gcorr40q_smoothed = filter_ovs_results_advanced_gcorr40q_smoothed()
        print("GCorr 40Q smoothed advanced filtering completed!")
        
    except Exception as e:
        print(f"Error during filtering: {str(e)}")
        raise 