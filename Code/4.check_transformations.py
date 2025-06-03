import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from scipy import stats
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroTransformer:
    def __init__(self, data_path, output_dir):
        self.data = pd.read_csv(data_path)
        self.data['yyyyqq'] = pd.to_datetime(self.data['yyyyqq'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define variables to analyze
        self.variables = [
            'FCPIQ', 'FGD$Q', 'FGDPL$Q', 'FGGDEBTGDPQ', 'FLBRQ',
            'FNETEXGSD$Q', 'FRGT10YQ', 'FSTOCKPQ', 'FTFXIUSAQ',
            'FCPWTI.IUSA'  # Added oil price
        ]
        
        # Define expected signs (None means no constraint)
        self.expected_signs = {
            'FCPIQ': 'positive',      # Inflation Rate
            'FGD$Q': None,            # GDP Level
            'FGDPL$Q': 'negative',    # GDP Growth
            'FGGDEBTGDPQ': 'positive', # Debt to GDP Ratio
            'FLBRQ': 'positive',      # Unemployment Rate
            'FNETEXGSD$Q': 'negative', # Net Exports
            'FRGT10YQ': 'positive',   # Interest Rate
            'FSTOCKPQ': None,         # Stock Price
            'FTFXIUSAQ': 'positive',  # Exchange Rate
            'FCPWTI.IUSA': None       # Oil Price (no sign constraint)
        }
        
        # Define variable groups and their expected transformations
        self.variable_groups = {
            'GDP': {
                'variables': ['FGD$Q', 'FGDPL$Q'],
                'expected_trans': 'log_return',  # GDP typically uses log returns
                'description': 'GDP and GDP per capita'
            },
            'Inflation': {
                'variables': ['FCPIQ'],
                'expected_trans': 'diff',  # Inflation typically uses differences
                'description': 'Consumer Price Index'
            },
            'Exchange_Rate': {
                'variables': ['FTFXIUSAQ'],
                'expected_trans': 'log_return',  # FX typically uses log returns
                'description': 'Exchange Rate vs USD'
            },
            'Interest_Rate': {
                'variables': ['FRGT10YQ'],
                'expected_trans': 'diff',  # Interest rates typically use differences
                'description': '10Y Government Bond Rate'
            },
            'Labor_Market': {
                'variables': ['FLBRQ'],
                'expected_trans': 'diff',  # Labor market indicators typically use differences
                'description': 'Labor Force Participation Rate'
            },
            'Fiscal': {
                'variables': ['FGGDEBTGDPQ', 'FNETEXGSD$Q'],
                'expected_trans': 'diff',  # Fiscal indicators typically use differences
                'description': 'Debt to GDP and Net Exports'
            },
            'Equity': {
                'variables': ['FSTOCKPQ'],
                'expected_trans': 'log_return',  # Equity typically uses log returns
                'description': 'Equity Market Index'
            },
            'Oil_Price': {
                'variables': ['FCPWTI.IUSA'],
                'expected_trans': 'log_return',
                'description': 'Oil Price (WTI)'
            }
        }
        
        # Initialize results storage
        self.transformation_decisions = {}
        self.transformed_data = None

    def remove_seasonality(self, series, period=4):
        """Remove seasonality from a time series using STL decomposition"""
        try:
            if series.isna().mean() > 0.5:
                return series, None

            # Create index for the full date range
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq='QE')
            series_regular = series.reindex(full_idx)
            
            # Handle missing values
            clean_series = series_regular.copy()
            clean_series = clean_series.interpolate(method='linear', limit=2)
            clean_series = clean_series.ffill().bfill()
            
            if len(clean_series.dropna()) < period * 2:
                return series, None
                
            decomposition = seasonal_decompose(clean_series, period=period, model='additive')
            
            adjusted = series.copy()
            mask = ~series.isna()
            if mask.any():
                seasonal_component = pd.Series(decomposition.seasonal, index=full_idx)
                seasonal_component = seasonal_component.reindex(series.index)
                adjusted[mask] = series[mask] - seasonal_component[mask]
            
            return adjusted, decomposition
        except Exception as e:
            logger.error(f"Error in seasonal adjustment: {str(e)}")
            return series, None

    def apply_transformation(self, series, trans_type):
        """Apply specified transformation to the series"""
        if trans_type == 'diff':
            return series.diff()
        elif trans_type == 'log_return':
            if (series > 0).all():
                return np.log(series).diff()
            else:
                logger.warning("Cannot apply log return to non-positive series")
                return series.diff()
        else:
            return series

    def check_stationarity(self, series, alpha=0.05):
        """Check stationarity using ADF test"""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 20:
                return {'is_stationary': False, 'p_value': np.nan, 'error': 'Insufficient data'}
            
            result = adfuller(clean_series, regression='ct')
            return {
                'is_stationary': result[1] < alpha,
                'p_value': result[1],
                'error': None
            }
        except Exception as e:
            return {'is_stationary': False, 'p_value': np.nan, 'error': str(e)}

    def analyze_variable_group(self, group_name, group_info):
        """Analyze a group of related variables to determine best transformation"""
        logger.info(f"\nAnalyzing {group_name} variables...")
        group_results = {
            'group_name': group_name,
            'description': group_info['description'],
            'variables': group_info['variables'],
            'expected_transformation': group_info['expected_trans'],
            'final_decision': None,
            'analysis': {}
        }
        
        for var in group_info['variables']:
            if var not in self.data.columns:
                logger.warning(f"Variable {var} not found in data")
                continue
                
            logger.info(f"Processing {var}...")
            var_results = {
                'stationarity_tests': {},
                'seasonality_tests': {},
                'distribution_tests': {}
            }
            
            # Analyze across all countries
            countries = self.data['cinc'].unique()
            stationarity_results = []
            seasonality_results = []
            
            # Create seasonal decomposition plots for each country
            for country in countries:
                country_data = self.data[self.data['cinc'] == country].copy()
                country_data = country_data.set_index('yyyyqq')
                series = country_data[var]
                
                # Check seasonality and create plots
                adjusted_series, decomp = self.remove_seasonality(series)
                if decomp is not None:
                    # Plot seasonal decomposition
                    try:
                        plt.figure(figsize=(12, 10))
                        decomp.plot()
                        plt.title(f'Seasonal Decomposition - {var} - {country}')
                        plt.tight_layout()
                        plt.savefig(self.output_dir / f'seasonal_decomp_{country}_{var}.png')
                        plt.close()
                        
                        seasonality_strength = np.std(decomp.seasonal) / np.std(decomp.trend)
                        seasonality_results.append(seasonality_strength)
                    except Exception as e:
                        logger.error(f"Error plotting seasonal decomposition for {var} - {country}: {str(e)}")
                        plt.close('all')
                
                # Apply transformations
                diff_series = self.apply_transformation(adjusted_series, 'diff')
                log_ret_series = self.apply_transformation(adjusted_series, 'log_return')
                
                # Test stationarity
                for trans_name, trans_series in [('original', adjusted_series), 
                                               ('diff', diff_series),
                                               ('log_return', log_ret_series if isinstance(log_ret_series, pd.Series) else None)]:
                    if trans_series is not None:
                        stat_test = self.check_stationarity(trans_series)
                        if stat_test['error'] is None:
                            stationarity_results.append({
                                'country': country,
                                'transformation': trans_name,
                                'is_stationary': stat_test['is_stationary'],
                                'p_value': stat_test['p_value']
                            })
            
            # Aggregate results
            if stationarity_results:
                stationarity_df = pd.DataFrame(stationarity_results)
                var_results['stationarity_tests'] = {
                    'stationary_ratio': stationarity_df.groupby('transformation')['is_stationary'].mean().to_dict(),
                    'mean_p_value': stationarity_df.groupby('transformation')['p_value'].mean().to_dict()
                }
            
            if seasonality_results:
                var_results['seasonality_tests'] = {
                    'mean_seasonality_strength': np.mean(seasonality_results),
                    'std_seasonality_strength': np.std(seasonality_results)
                }
            
            # Make transformation decision
            if var_results['stationarity_tests']:
                stationary_ratios = var_results['stationarity_tests']['stationary_ratio']
                best_trans = max(stationary_ratios.items(), key=lambda x: x[1])[0]
                var_results['recommended_transformation'] = best_trans
            else:
                var_results['recommended_transformation'] = group_info['expected_trans']
            
            group_results['analysis'][var] = var_results
        
        # Make final group decision
        recommended_trans = {}
        for var, analysis in group_results['analysis'].items():
            recommended_trans[var] = analysis['recommended_transformation']
        
        # Use most common transformation as group decision
        group_results['final_decision'] = max(set(recommended_trans.values()), key=list(recommended_trans.values()).count)
        
        return group_results

    def transform_data(self):
        """Transform the data according to the decisions"""
        logger.info("\nTransforming data according to decisions...")
        transformed_data = self.data.copy()
        
        for group_name, group_info in self.variable_groups.items():
            trans_type = self.transformation_decisions[group_name]['final_decision']
            for var in group_info['variables']:
                if var in transformed_data.columns:
                    # First remove seasonality
                    for country in transformed_data['cinc'].unique():
                        mask = transformed_data['cinc'] == country
                        series = transformed_data.loc[mask, var]
                        adjusted_series, _ = self.remove_seasonality(series)
                        transformed_data.loc[mask, var] = adjusted_series
                    
                    # Then apply transformation
                    transformed_data[f'{var}_trans'] = self.apply_transformation(
                        transformed_data[var], 
                        trans_type
                    )
        
        self.transformed_data = transformed_data
        return transformed_data

    def plot_transformation_results(self, group_name, group_info):
        """Plot transformation results for a variable group"""
        logger.info(f"\nPlotting results for {group_name}...")
        for var in group_info['variables']:
            if var not in self.data.columns:
                continue
                
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Original vs Seasonally Adjusted
            plt.subplot(2, 2, 1)
            for country in self.data['cinc'].unique()[:5]:  # Plot first 5 countries
                country_data = self.data[self.data['cinc'] == country]
                plt.plot(country_data['yyyyqq'], country_data[var], label=country, alpha=0.5)
            plt.title(f'{var} - Original Data (Sample Countries)')
            plt.xticks(rotation=45)
            plt.legend()
            
            # Plot 2: Transformed Data
            plt.subplot(2, 2, 2)
            trans_var = f'{var}_trans'
            if trans_var in self.transformed_data.columns:
                for country in self.transformed_data['cinc'].unique()[:5]:
                    country_data = self.transformed_data[self.transformed_data['cinc'] == country]
                    plt.plot(country_data['yyyyqq'], country_data[trans_var], label=country, alpha=0.5)
                plt.title(f'{var} - Transformed Data (Sample Countries)')
                plt.xticks(rotation=45)
                plt.legend()
            
            # Plot 3: Distribution of Transformed Data
            plt.subplot(2, 2, 3)
            if trans_var in self.transformed_data.columns:
                sns.histplot(data=self.transformed_data, x=trans_var, bins=50)
                plt.title(f'{var} - Distribution of Transformed Data')
            
            # Plot 4: QQ Plot
            plt.subplot(2, 2, 4)
            if trans_var in self.transformed_data.columns:
                stats.probplot(self.transformed_data[trans_var].dropna(), dist="norm", plot=plt)
                plt.title(f'{var} - QQ Plot')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'transformation_{var}.png')
            plt.close()

    def plot_pattern_overlay(self, var):
        """Plot original and transformed series for a sample country (first country with non-null data)"""
        sample_country = None
        for country in self.data['cinc'].unique():
            if not self.data[self.data['cinc'] == country][var].isnull().all():
                sample_country = country
                break
        if sample_country is None:
            return
        orig = self.data[self.data['cinc'] == sample_country].set_index('yyyyqq')[var]
        trans_var = f'{var}_trans'
        if trans_var not in self.transformed_data.columns:
            return
        trans = self.transformed_data[self.transformed_data['cinc'] == sample_country].set_index('yyyyqq')[trans_var]
        plt.figure(figsize=(12, 6))
        plt.plot(orig.index, orig, label='Original', alpha=0.7)
        plt.plot(trans.index, trans, label='Transformed', alpha=0.7)
        plt.title(f'{var}: Original vs Transformed ({sample_country})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'pattern_{var}.png')
        plt.close()

    def save_transformation_summary(self):
        """Save a high-level summary of transformation decisions by macro variable type"""
        summary_lines = [
            "Macro Variable Transformation Summary",
            "===================================",
            "",
            "This document summarizes the transformation decisions made for each macro variable type.",
            "The transformations were chosen based on stationarity tests and economic theory.",
            "",
            "Transformation Types:",
            "- diff: First difference (x_t - x_{t-1})",
            "- log_return: Log return (log(x_t/x_{t-1}))",
            "",
            "Summary by Variable Type:",
            "------------------------"
        ]
        
        for group_name, decision in self.transformation_decisions.items():
            summary_lines.extend([
                f"\n{group_name}:",
                f"Description: {decision['description']}",
                f"Final transformation: {decision['final_decision']}",
                "Variables:"
            ])
            for var, analysis in decision['analysis'].items():
                summary_lines.append(f"  - {var}: {analysis['recommended_transformation']}")
        
        # Add timestamp
        from datetime import datetime
        summary_lines.extend([
            "",
            f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Note: All variables have been seasonally adjusted using STL decomposition before transformation."
        ])
        
        # Save summary
        summary_file = self.output_dir / 'transformation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        logger.info(f"Saved transformation summary to {summary_file}")

    def save_stationarity_results(self):
        """Save stationarity test results for all final transformed macro variables by country"""
        logger.info("Saving stationarity test results for all final transformed macro variables...")
        results = []
        for group_name, group_info in self.variable_groups.items():
            trans_type = self.transformation_decisions[group_name]['final_decision']
            for var in group_info['variables']:
                trans_var = f'{var}_trans'
                if trans_var in self.transformed_data.columns:
                    for country in self.transformed_data['cinc'].unique():
                        series = self.transformed_data.loc[self.transformed_data['cinc'] == country, trans_var]
                        stat = self.check_stationarity(series)
                        results.append({
                            'variable': trans_var,
                            'country': country,
                            'is_stationary': stat['is_stationary'],
                            'p_value': stat['p_value'],
                            'error': stat['error']
                        })
        import pandas as pd
        df = pd.DataFrame(results)
        out_path = self.output_dir / 'stationarity_results_transformed.csv'
        df.to_csv(out_path, index=False)
        logger.info(f"Saved stationarity results to {out_path}")

    def run_analysis(self):
        """Run the complete transformation analysis"""
        try:
            logger.info("Starting transformation analysis...")
            
            # Analyze each variable group
            for group_name, group_info in self.variable_groups.items():
                group_results = self.analyze_variable_group(group_name, group_info)
                self.transformation_decisions[group_name] = group_results
            
            # Save transformation decisions
            decisions_file = self.output_dir / 'transformation_decisions.json'
            with open(decisions_file, 'w') as f:
                json.dump(self.transformation_decisions, f, indent=2)
            logger.info(f"Saved transformation decisions to {decisions_file}")
            
            # Save high-level summary
            self.save_transformation_summary()
            
            # Transform the data
            transformed_data = self.transform_data()
            
            # Save transformed data
            transformed_data.to_csv(self.output_dir / 'transformed_data.csv', index=False)
            logger.info("Saved transformed data")
            
            # Save stationarity results for all final transformed macro variables
            self.save_stationarity_results()
            
            # Create plots
            for group_name, group_info in self.variable_groups.items():
                self.plot_transformation_results(group_name, group_info)
                # Extra: overlay plot for each variable
                for var in group_info['variables']:
                    self.plot_pattern_overlay(var)
            
            # Print summary
            logger.info("\nTransformation Analysis Summary:")
            for group_name, decision in self.transformation_decisions.items():
                logger.info(f"\n{group_name}:")
                logger.info(f"Description: {decision['description']}")
                logger.info(f"Final transformation: {decision['final_decision']}")
                logger.info("Variables:")
                for var, analysis in decision['analysis'].items():
                    logger.info(f"  {var}: {analysis['recommended_transformation']}")
            
            return self.transformation_decisions, transformed_data
            
        except Exception as e:
            logger.error(f"Error in transformation analysis: {str(e)}", exc_info=True)
            raise

def main():
    try:
        # Initialize transformer
        transformer = MacroTransformer(
            data_path='Output/3.regression_data.csv',
            output_dir='Output/4.transformation'
        )
        
        # Run analysis
        decisions, transformed_data = transformer.run_analysis()
        
        logger.info("\nAnalysis complete!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        plt.close('all')

if __name__ == "__main__":
    main() 