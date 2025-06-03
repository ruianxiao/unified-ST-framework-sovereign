import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Output directories
output_dir = Path('Output/1.2.sov_edf_assessment')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
edf = pd.read_csv('Output/1.sov_use_052025.csv')
edf['price_date'] = pd.to_datetime(edf['price_date'])

# 1. Data Overview
summary = {}
summary['n_countries'] = edf['cinc'].nunique()
summary['n_obs'] = len(edf)
summary['date_min'] = edf['price_date'].min()
summary['date_max'] = edf['price_date'].max()
summary['years'] = summary['date_max'].year - summary['date_min'].year + 1

# Geographic distribution
group_cols = ['region', 'cinc', 'entityName']
geo_counts = edf.groupby(group_cols).size().reset_index(name='n_obs')
geo_summary = geo_counts.groupby('region').agg({'cinc':'nunique', 'n_obs':'sum'}).reset_index()
geo_summary.rename(columns={'cinc':'n_countries'}, inplace=True)
geo_summary.to_csv(output_dir/'geo_summary.csv', index=False)

# 2. Compare 1y vs 5y EDF
if 'cdsiedf1' in edf.columns and 'cdsiedf5' in edf.columns:
    # Scatter plot
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='cdsiedf1', y='cdsiedf5', data=edf, alpha=0.3)
    plt.title('1Y vs 5Y CDS-Implied EDF')
    plt.xlabel('1Y EDF')
    plt.ylabel('5Y EDF')
    plt.savefig(output_dir/'edf1_vs_edf5_scatter.png')
    plt.close()
    # Correlation
    corr = edf[['cdsiedf1','cdsiedf5']].corr().iloc[0,1]
    # Per-country time series plot
    for country in edf['cinc'].unique():
        sub = edf[edf['cinc']==country]
        plt.figure(figsize=(10,4))
        plt.plot(sub['price_date'], sub['cdsiedf1'], label='1Y')
        plt.plot(sub['price_date'], sub['cdsiedf5'], label='5Y')
        plt.title(f'EDF Time Series: {country}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir/f'edf_timeseries_{country}.png')
        plt.close()
    # Summary stats
    edf_stats = edf[['cdsiedf1','cdsiedf5']].describe().T
    edf_stats['missing'] = edf[['cdsiedf1','cdsiedf5']].isna().sum()
    edf_stats.to_csv(output_dir/'edf_stats.csv')
else:
    corr = np.nan
    edf_stats = pd.DataFrame()

# 3. Data Quality Assessment
# Missingness by country
missing_by_country = edf.groupby('cinc')[['cdsiedf5']].apply(lambda x: x.isna().mean()).reset_index()
missing_by_country.rename(columns={'cdsiedf5':'missing_frac'}, inplace=True)
missing_by_country.to_csv(output_dir/'missing_by_country.csv', index=False)

# Outlier detection (5Y EDF)
edf['outlier'] = (edf['cdsiedf5'] > edf['cdsiedf5'].quantile(0.99)) | (edf['cdsiedf5'] < edf['cdsiedf5'].quantile(0.01))

# 4. Quarterly Aggregation
edf['yyyyqq'] = edf['price_date'] + pd.offsets.QuarterEnd(0)
edf_q = edf.groupby(['cinc','yyyyqq']).agg({'cdsiedf5':'mean'}).reset_index()

# Extreme value assessment (global percentiles)
global_99 = edf['cdsiedf5'].quantile(0.99)
global_01 = edf['cdsiedf5'].quantile(0.01)

# For each country, check data suitability and extreme values
country_suitability = []
for country, sub in edf_q.groupby('cinc'):
    # Generate full range of quarters
    min_q = sub['yyyyqq'].min()
    max_q = sub['yyyyqq'].max()
    all_quarters = pd.date_range(min_q, max_q, freq='Q')
    sub_full = sub.set_index('yyyyqq').reindex(all_quarters)
    n_quarters = len(all_quarters)
    n_missing = sub_full['cdsiedf5'].isna().sum()
    frac_missing = n_missing / n_quarters
    # Extreme value assessment
    n_extremes = ((sub_full['cdsiedf5'] > global_99) | (sub_full['cdsiedf5'] < global_01)).sum()
    extreme_flag = n_extremes > 0
    suitable = (n_quarters >= 20) and (frac_missing < 0.2)
    reason = []
    if n_quarters < 20:
        reason.append('too few quarters')
    if frac_missing >= 0.2:
        reason.append('too much missing')
    if extreme_flag:
        reason.append('extreme values')
    country_suitability.append({
        'cinc': country,
        'n_quarters': n_quarters,
        'frac_missing': frac_missing,
        'n_extremes': n_extremes,
        'extreme_flag': extreme_flag,
        'reason': '; '.join(reason) if reason else 'suitable'
    })
suitability_df = pd.DataFrame(country_suitability)
suitability_df.to_csv(output_dir/'country_suitability.csv', index=False)

# Save summary for RMarkdown
summary_df = pd.DataFrame([summary])
summary_df.to_csv(output_dir/'summary.csv', index=False)

# Save correlation
with open(output_dir/'edf1_edf5_corr.txt','w') as f:
    f.write(f'Correlation between 1Y and 5Y EDF: {corr:.3f}\n')

# Save outlier info
edf[['cinc','price_date','cdsiedf5','outlier']].to_csv(output_dir/'edf_outliers.csv', index=False)

print('Analysis complete. Results saved in Output/1.2.sov_edf_assessment/') 