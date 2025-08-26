# Sovereign Probability of Default Modeling Framework

This repository contains the code for a unified short-term framework for sovereign probability of default (PD) modeling. The framework processes sovereign CDS data and macroeconomic variables to generate PD estimates and perform regression analysis.

## Project Structure

- `Code/`: Contains all Python scripts and notebooks for data processing and analysis
  - `1.1.prepare_cdsiedf_data.ipynb`: Jupyter notebook for CDS implied EDF data preparation
  - `1.2.analyze_sov_edf.py`: Analysis of sovereign EDF data
  - `1.2.sov_edf_data_assessment_pptx.py`: PowerPoint generation for EDF data assessment
  - `2.prepare_mv_data.ipynb`: Jupyter notebook for macroeconomic variable data preparation
  - `3.prepare_regression_data.py`: Prepares data for regression analysis
  - `4.check_transformations.py`: Validates data transformations
  - `6.ovs_variable_selection.py`: Out-of-sample variable selection analysis
  - `7.filter_ovs_results.py`: Filters and processes OVS results
  - `8.backtesting_analysis.py`: Performs backtesting analysis
  - `8.scenario_forecast.py`: Generates scenario-based forecasts
  - `8.1.scenario_forecast_top10.py`: Generates forecasts for top 10 models
  - `9.compare_ovs_gcorr_forecasts.py`: Compares OVS and GCorr forecast results
  - `9.1.compare_ovs_gcorr_top10_models.py`: Compares top 10 OVS and GCorr models
  - `10.compare_backtesting_results.py`: Compares backtesting results across models

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ruianxiao/unified-ST-framework-sovereign.git
cd unified-ST-framework-sovereign
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Requirements

The framework requires the following input data:
- Sovereign CDS data
- Macroeconomic variables
- Country mappings and mnemonics

**Note**: 
- Input data files are not included in this repository due to confidentiality.
- This repository only tracks the `Code/` folder containing the analysis scripts and notebooks. All other folders and files are excluded via `.gitignore` for security and confidentiality purposes.

## Usage

The scripts should be run in sequence:

1. Prepare CDS implied EDF data:
```bash
# Open and run the Jupyter notebook
jupyter notebook Code/1.1.prepare_cdsiedf_data.ipynb
```

2. Analyze sovereign EDF data:
```bash
python Code/1.2.analyze_sov_edf.py
```

3. Prepare macroeconomic variables:
```bash
# Open and run the Jupyter notebook
jupyter notebook Code/2.prepare_mv_data.ipynb
```

4. Prepare regression data:
```bash
python Code/3.prepare_regression_data.py
```

5. Check data transformations:
```bash
python Code/4.check_transformations.py
```

6. Perform variable selection analysis:
```bash
python Code/6.ovs_variable_selection.py
```

7. Filter OVS results:
```bash
python Code/7.filter_ovs_results.py
```

8. Run backtesting and scenario analysis:
```bash
python Code/8.backtesting_analysis.py
python Code/8.scenario_forecast.py
```

9. Compare models and results:
```bash
python Code/9.compare_ovs_gcorr_forecasts.py
python Code/10.compare_backtesting_results.py
```

## Output

The framework generates:
- Processed CDS implied EDF data
- Processed macroeconomic variables
- Variable selection and regression analysis results
- Backtesting analysis and model comparison results
- Scenario-based forecasts and projections
- Statistical analysis and visualizations
- Automated presentation slides and reports

## License

This project is proprietary and confidential. All rights reserved.

## Contact

For questions or support, please contact the repository owner. 