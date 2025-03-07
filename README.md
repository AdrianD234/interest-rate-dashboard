# Advanced Interest Rate & Revenue Dashboard

A sophisticated, interactive dashboard for analyzing the relationship between interest rates and revenue inflows, with advanced visualization and analysis capabilities.

![Dashboard Preview](dashboard_preview.png)

## Features

### Core Analysis Features
- **Multi-lag Analysis**: Test relationships between interest rates and revenue with configurable time lags (T+0 to T+12)
- **Pre/Post COVID Analysis**: Filter data to compare relationships before and after COVID onset
- **Multiple Interest Rate Variables**: Compare using OCR, 30-day bill rates, or changes in rates
- **Statistical Analysis**: View comprehensive regression statistics including R-squared, p-values, coefficients

### Advanced Visualization
- **Interactive Time Series**: Advanced time series charts with multiple overlays and toggleable options
- **Correlation Heatmap**: Visualize correlation strength across different time lags
- **Distribution Analysis**: View histograms with density curves for both variables
- **Seasonal Decomposition**: Break down time series into trend, seasonal, and residual components
- **Statistical Forecasting**: ARIMA forecasting with confidence intervals (requires pmdarima)

### Interactive UI Elements
- **Key Metrics Cards**: At-a-glance summary of important statistics
- **Advanced Data Grid**: Sortable, filterable table with custom formatting
- **Export Options**: Download data in CSV or Excel formats
- **Responsive Design**: Works on both desktop and mobile devices

## Installation

1. Clone this repository or download the files to your local machine
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your `Rev_IR.xlsx` file is in the same directory as the app files
2. Run the standard or enhanced app:

```bash
# Run the standard dashboard
python app.py

# Run the enhanced dashboard with advanced features
python enhanced_app.py
```

3. Open your web browser and navigate to http://127.0.0.1:8050/

## Two Dashboard Versions

This repository includes two dashboard implementations:

1. **Standard Dashboard (`app.py`)**: Core functionality with basic styling, works with minimal dependencies
2. **Enhanced Dashboard (`enhanced_app.py`)**: Premium experience with advanced visualization, forecasting, and UI elements

The enhanced dashboard automatically falls back to simpler components if advanced libraries aren't available.

## Advanced Features

### Additional Libraries

For the full enhanced experience, install these optional packages:

```bash
pip install dash-daq dash-extensions dash-mantine-components dash-cytoscape dash-ag-grid plotly-resampler pmdarima ppscore kaleido
```

### Data Processing

- Automatic handling of various input formats
- Built-in error handling and recovery
- Intelligent column mapping and data conversion

### Analytical Methods

- Multiple correlation metrics including Pearson and Predictive Power Score
- Time series analysis with seasonal decomposition
- ARIMA forecasting with automatic parameter selection
- Confidence intervals for regression analysis

## Customization

The dashboard is designed to be easily customizable:

- Color scheme can be modified in the COLOR_SCHEME dictionary
- Additional variables can be added to the analysis
- New visualization types can be incorporated
- UI components can be rearranged or styled differently

## Requirements

- Python 3.7+
- Core dependencies: pandas, numpy, plotly, dash, statsmodels, scikit-learn
- Optional advanced dependencies: listed in requirements.txt