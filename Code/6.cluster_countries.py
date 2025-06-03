import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SovereignClusterer:
    """
    Knowledge-based sovereign clustering that groups countries based on:
    - Economic development and market access
    - Regional economic integration
    - Financial market characteristics
    - Natural resource dependence
    - Political and institutional factors
    """
    def __init__(self, data_path: str = 'Output/4.transformed_data.csv'):
        self.data = pd.read_csv(data_path)
        self.data['yyyyqq'] = pd.to_datetime(self.data['yyyyqq'])
        
        # Define sovereign clusters based on economic and financial characteristics
        self.sovereign_clusters = {
            'ADVANCED_ECONOMIES': {
                'description': 'High-income economies with deep financial markets and strong institutions',
                'countries': ['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'JPN', 'AUS', 'CHE', 'SWE', 
                            'NLD', 'BEL', 'AUT', 'DNK', 'NOR', 'FIN', 'NZL', 'SGP', 'HKG', 'KOR'],
                'characteristics': {
                    'market_access': 'High',
                    'financial_depth': 'Deep',
                    'institutional_quality': 'Strong',
                    'economic_diversification': 'High'
                }
            },
            'EURO_PERIPHERY': {
                'description': 'Eurozone countries with higher sovereign risk and structural challenges',
                'countries': ['GRC', 'PRT', 'ESP', 'IRL', 'CYP'],
                'characteristics': {
                    'market_access': 'Moderate',
                    'financial_depth': 'Moderate',
                    'institutional_quality': 'Moderate',
                    'economic_diversification': 'Moderate'
                }
            },
            'EM_ASIA': {
                'description': 'Dynamic emerging Asian economies with strong growth potential',
                'countries': ['CHN', 'IDN', 'MYS', 'THA', 'PHL', 'VNM', 'IND'],
                'characteristics': {
                    'market_access': 'Moderate',
                    'financial_depth': 'Growing',
                    'institutional_quality': 'Improving',
                    'economic_diversification': 'Moderate'
                }
            },
            'LATAM_MAJORS': {
                'description': 'Major Latin American economies with commodity exposure',
                'countries': ['BRA', 'MEX', 'CHL', 'COL', 'PER', 'ARG'],
                'characteristics': {
                    'market_access': 'Moderate',
                    'financial_depth': 'Moderate',
                    'institutional_quality': 'Moderate',
                    'economic_diversification': 'Moderate'
                }
            },
            'GCC': {
                'description': 'Gulf Cooperation Council countries with hydrocarbon dependence',
                'countries': ['SAU', 'UAE', 'QAT', 'KWT', 'BHR', 'OMN'],
                'characteristics': {
                    'market_access': 'High',
                    'financial_depth': 'Moderate',
                    'institutional_quality': 'Moderate',
                    'economic_diversification': 'Low'
                }
            },
            'EASTERN_EUROPE': {
                'description': 'Eastern European economies with EU integration',
                'countries': ['POL', 'CZE', 'HUN', 'ROU', 'BGR', 'HRV', 'SVK', 'SVN', 'EST', 'LVA', 'LTU'],
                'characteristics': {
                    'market_access': 'Moderate',
                    'financial_depth': 'Growing',
                    'institutional_quality': 'Improving',
                    'economic_diversification': 'Moderate'
                }
            },
            'FRONTIER_MARKETS': {
                'description': 'Frontier markets with developing financial systems',
                'countries': ['EGY', 'MAR', 'TUN', 'KEN', 'NGA', 'PAK', 'BGD', 'LKA'],
                'characteristics': {
                    'market_access': 'Low',
                    'financial_depth': 'Shallow',
                    'institutional_quality': 'Weak',
                    'economic_diversification': 'Low'
                }
            }
        }
        
        # Define economic indicators for analysis
        self.economic_indicators = {
            'macro_stability': ['FGDPL$Q', 'FCPIQ'],  # GDP growth, inflation
            'fiscal_health': ['FGGDEBTGDPQ', 'FNETEXGSD$Q'],  # Debt/GDP, trade balance
            'external_vulnerability': ['FTFXIUSAQ', 'FLBRQ'],  # FX reserves, labor market
            'market_risk': ['dlnPD', 'FRGT10YQ']  # PD volatility, interest rates
        }

    def assign_clusters(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Assign countries to clusters based on knowledge-based classification
        """
        logger.info("Assigning countries to sovereign clusters...")
        
        # Create cluster assignments
        cluster_assignments = []
        for cluster_name, cluster_info in self.sovereign_clusters.items():
            for country in cluster_info['countries']:
                if country in self.data['cinc'].unique():
                    cluster_assignments.append({
                        'cinc': country,
                        'cluster': cluster_name,
                        'cluster_description': cluster_info['description']
                    })
        
        assignments_df = pd.DataFrame(cluster_assignments)
        
        # Calculate cluster characteristics
        cluster_characteristics = {}
        for cluster_name, cluster_info in self.sovereign_clusters.items():
            cluster_countries = assignments_df[assignments_df['cluster'] == cluster_name]['cinc'].tolist()
            if not cluster_countries:
                continue
                
            # Calculate economic indicators for the cluster
            cluster_data = self.data[self.data['cinc'].isin(cluster_countries)]
            
            characteristics = {
                'countries': cluster_countries,
                'size': len(cluster_countries),
                'description': cluster_info['description'],
                'characteristics': cluster_info['characteristics'],
                'economic_indicators': {}
            }
            
            # Calculate average economic indicators
            for category, indicators in self.economic_indicators.items():
                characteristics['economic_indicators'][category] = {}
                for indicator in indicators:
                    if indicator in cluster_data.columns:
                        mean_value = cluster_data[indicator].mean()
                        std_value = cluster_data[indicator].std()
                        characteristics['economic_indicators'][category][indicator] = {
                            'mean': mean_value,
                            'std': std_value
                        }
            
            cluster_characteristics[cluster_name] = characteristics
        
        return assignments_df, cluster_characteristics

    def visualize_clusters(self, assignments: pd.DataFrame, characteristics: Dict):
        """
        Create visualizations for the sovereign clusters
        """
        output_dir = Path('Output/6.clustering')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create cluster size visualization
        plt.figure(figsize=(12, 6))
        cluster_sizes = assignments.groupby('cluster').size()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
        plt.title('Sovereign Cluster Sizes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / '6.cluster_sizes.png')
        plt.close()
        
        # 2. Create economic indicator heatmap
        indicator_data = []
        for cluster_name, chars in characteristics.items():
            for category, indicators in chars['economic_indicators'].items():
                for indicator, values in indicators.items():
                    indicator_data.append({
                        'Cluster': cluster_name,
                        'Category': category,
                        'Indicator': indicator,
                        'Mean': values['mean'],
                        'Std': values['std']
                    })
        
        indicator_df = pd.DataFrame(indicator_data)
        plt.figure(figsize=(15, 10))
        pivot_data = indicator_df.pivot_table(
            values='Mean',
            index='Cluster',
            columns=['Category', 'Indicator'],
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Cluster Economic Indicators')
        plt.tight_layout()
        plt.savefig(output_dir / '6.cluster_indicators.png')
        plt.close()
        
        # 3. Save detailed cluster information
        with open(output_dir / '6.cluster_details.txt', 'w') as f:
            for cluster_name, chars in characteristics.items():
                f.write(f"\n{cluster_name}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Description: {chars['description']}\n")
                f.write(f"Countries: {', '.join(chars['countries'])}\n")
                f.write(f"Size: {chars['size']}\n\n")
                
                f.write("Characteristics:\n")
                for char, value in chars['characteristics'].items():
                    f.write(f"  {char}: {value}\n")
                
                f.write("\nEconomic Indicators:\n")
                for category, indicators in chars['economic_indicators'].items():
                    f.write(f"\n  {category}:\n")
                    for indicator, values in indicators.items():
                        f.write(f"    {indicator}:\n")
                        f.write(f"      Mean: {values['mean']:.4f}\n")
                        f.write(f"      Std:  {values['std']:.4f}\n")
        
        # 4. Save cluster assignments
        assignments.to_csv(output_dir / '6.cluster_assignments.csv', index=False)

def main():
    logger.info("Starting sovereign clustering analysis...")
    clusterer = SovereignClusterer()
    assignments, characteristics = clusterer.assign_clusters()
    clusterer.visualize_clusters(assignments, characteristics)
    logger.info("Clustering analysis complete. Results saved in Output/6.clustering/")

if __name__ == "__main__":
    main() 