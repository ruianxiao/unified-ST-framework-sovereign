# Sovereign Probability of Default Modeling Framework

This repository contains the code for a unified short-term framework for sovereign probability of default (PD) modeling. The framework processes sovereign CDS data and macroeconomic variables to generate PD estimates and perform regression analysis.

## Project Structure

- `Code/`: Contains all Python scripts for data processing and analysis
  - `1.process_sov_pd.py`: Processes sovereign PD data
  - `2.process_macro_variables.py`: Processes macroeconomic variables
  - `3.prepare_regression_data.py`: Prepares data for regression analysis
  - `4.run_regression.py`: Performs regression analysis
  - `5.analyze_regression_results.py`: Analyzes regression results
  - `generate_pptx.py`: Generates presentation slides
  - `generate_pptx_formatted.py`: Generates formatted presentation slides

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

Note: Input data files are not included in this repository due to confidentiality.

## Usage

The scripts should be run in sequence:

1. Process sovereign PD data:
```bash
python Code/1.process_sov_pd.py
```

2. Process macroeconomic variables:
```bash
python Code/2.process_macro_variables.py
```

3. Prepare regression data:
```bash
python Code/3.prepare_regression_data.py
```

4. Run regression analysis:
```bash
python Code/4.run_regression.py
```

5. Analyze results:
```bash
python Code/5.analyze_regression_results.py
```

## Output

The framework generates:
- Processed PD data
- Processed macroeconomic variables
- Regression analysis results
- Statistical analysis and visualizations
- Presentation slides

## License

This project is proprietary and confidential. All rights reserved.

## Contact

For questions or support, please contact the repository owner. 