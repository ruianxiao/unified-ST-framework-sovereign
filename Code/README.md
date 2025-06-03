# Sovereign Research Code

This repository contains research code for analyzing sovereign credit risk using CDS (Credit Default Swaps) and macroeconomic variables.

## Project Structure

```
.
├── Code/
│   └── backtesting.R         # R script for backtesting analysis
├── 1.prepare_cdsiedf_data.ipynb  # Notebook for processing CDS implied EDF data
└── 2.prepare_mv_data.ipynb       # Notebook for preparing macroeconomic variables
```

## Data Processing Pipeline

1. **CDS Data Preparation** (`1.prepare_cdsiedf_data.ipynb`):
   - Downloads raw CDS data from databases
   - Combines data from CMA and Credit Edge datasets
   - Performs liquidity filtering
   - Generates monthly CDS implied EDF data

2. **Macroeconomic Variables** (`2.prepare_mv_data.ipynb`):
   - Processes various economic indicators including:
     - GDP
     - Unemployment Rate
     - Exchange Rates
     - Consumer Price Index
     - Government Bond Yields
     - External Debt metrics

3. **Backtesting** (`Code/backtesting.R`):
   - Performs backtesting analysis on the processed data

## Dependencies

### Python Dependencies
- pandas
- numpy
- pyodbc
- matplotlib
- statsmodels
- scipy

### R Dependencies
- Required R packages are listed in the backtesting.R script

## Output Structure

The code generates outputs in the following directories:
- `Output/`: Processed data files
- `Plot/`: Generated visualizations and charts

## Notes

- Data sources include both CMA and Markit
- The project uses a quarterly and monthly data frequency
- Sovereign data includes various economic indicators and CDS metrics 