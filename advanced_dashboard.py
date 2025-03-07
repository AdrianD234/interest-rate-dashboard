import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set Plotly template to a modern, visually appealing style
pio.templates.default = "plotly_white"

# Define custom color scheme - more professional palette
COLOR_SCHEME = {
    'primary': '#0A3D62',       # Dark blue - primary brand
    'secondary': '#3282B8',     # Medium blue - secondary brand
    'accent': '#F39C12',        # Amber - accent
    'background': '#F5F8FA',    # Light gray - background
    'text': '#1A1A2E',          # Dark navy - text
    'grid': '#DAE1E7',          # Light gray - grid lines
    'revenue': '#0E7C7B',       # Teal - revenue
    'interest': '#6A2C70',      # Purple - interest rate
    'correlation_positive': '#1E8449', # Green - positive correlation
    'correlation_negative': '#922B21', # Red - negative correlation
    'neutral': '#566573',       # Slate - neutral
    'highlight': '#1289A7',     # Highlight color
    'widget_bg': '#FFFFFF',     # Widget background
    'positive': '#27AE60',      # Positive values
    'negative': '#E74C3C',      # Negative values
    'warning': '#F39C12',       # Warning
    'info': '#3498DB'           # Information
}

# Load the data
df = pd.read_excel('Rev_IR.xlsx')

# Print column names to debug
print("Column names in Excel file:", df.columns.tolist())

# Handle column mapping safely
try:
    # Try standard column names first
    if 'Date' in df.columns:
        # Columns are already named correctly
        pass
    else:
        # Rename based on position (assuming the structure you shared)
        column_mapping = {}
        
        # Map at least 6 columns, if available
        expected_columns = ['Date', 'Total', 'OCR', '30 day bill', 'Δ Revenue', 'Δ 30 day']
        for i, expected_col in enumerate(expected_columns):
            if i < len(df.columns):
                column_mapping[df.columns[i]] = expected_col
                
        print(f"Mapping columns: {column_mapping}")
        df = df.rename(columns=column_mapping)
        
    # If we're still missing Δ Revenue or Δ 30 day, calculate them
    if 'Total' in df.columns and 'Δ Revenue' not in df.columns:
        print("Calculating Δ Revenue")
        df['Δ Revenue'] = df['Total'].pct_change()
        
    if '30 day bill' in df.columns and 'Δ 30 day' not in df.columns:
        print("Calculating Δ 30 day")
        df['Δ 30 day'] = df['30 day bill'].pct_change()
    
    # Convert percentage strings to float values if needed
    for col in ['OCR', '30 day bill', 'Δ Revenue', 'Δ 30 day']:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert percentage strings (like '2.5%') to float (0.025)
                try:
                    df[col] = df[col].str.rstrip('%').astype('float') / 100
                    print(f"Converted {col} from percentage string to float")
                except:
                    print(f"Could not convert {col} from string to float")
            elif df[col].max() > 1 and col in ['OCR', '30 day bill']:
                # If values are like 2.5 (percent) instead of 0.025 (decimal)
                df[col] = df[col] / 100
                print(f"Converted {col} from percentage to decimal")
        
    print("Final columns:", df.columns.tolist())
except Exception as e:
    print(f"Error during column mapping: {e}")
    # Create minimal mock dataset if all else fails
    if len(df.columns) < 6:
        print("WARNING: Creating mock data for missing columns")
        if df.columns[0] != 'Date':
            df = df.rename(columns={df.columns[0]: 'Date'})
        if len(df.columns) <= 1 or df.columns[1] != 'Total':
            df['Total'] = 100000000
        if 'OCR' not in df.columns:
            df['OCR'] = 0.025
        if '30 day bill' not in df.columns:
            df['30 day bill'] = 0.026
        if 'Δ Revenue' not in df.columns:
            df['Δ Revenue'] = 0.0
        if 'Δ 30 day' not in df.columns:
            df['Δ 30 day'] = 0.0

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add useful derived columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')

# Calculate additional metrics
df['Revenue_MA_3M'] = df['Total'].rolling(window=3).mean()
df['Revenue_MA_12M'] = df['Total'].rolling(window=12).mean()
df['Revenue_YoY'] = df.groupby(df['Date'].dt.month)['Total'].pct_change(12)

# Set time periods for COVID analysis
COVID_START = pd.to_datetime('2020-03-01')  # March 2020 as approximate COVID start

# Create Δ Revenue time lags for all months and years
print("Creating lag variables...")

# Monthly lags (0-12 months)
df['Δ Revenue_T0'] = df['Δ Revenue']  # No lag
for lag in range(1, 13):
    df[f'Δ Revenue_M{lag}'] = df['Δ Revenue'].shift(-lag)
    print(f"Created lag: Δ Revenue_M{lag}")

# Yearly lags (1-3 years)
# Use specific monthly lags for yearly data to avoid confusion
df['Δ Revenue_Y1'] = df['Δ Revenue_M12']  # 1 year = 12 months
df['Δ Revenue_Y2'] = df['Δ Revenue'].shift(-24)  # 2 years = 24 months
df['Δ Revenue_Y3'] = df['Δ Revenue'].shift(-36)  # 3 years = 36 months

print("Lag variables created:")

# Find the min and max dates for the time slider
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = [min_date, max_date]

# Create the Dash app with a more professional theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Interest Rate & Revenue Analytics Platform"

# Custom CSS for more professional styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
                background-color: ''' + COLOR_SCHEME['background'] + ''';
                color: ''' + COLOR_SCHEME['text'] + ''';
            }
            .card {
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                border: none;
            }
            .card-header {
                background-color: ''' + COLOR_SCHEME['primary'] + ''';
                color: white;
                border-top-left-radius: 8px !important;
                border-top-right-radius: 8px !important;
                font-weight: 500;
                padding: 12px 20px;
            }
            .card-body {
                padding: 20px;
                background-color: ''' + COLOR_SCHEME['widget_bg'] + ''';
            }
            .dash-dropdown .Select-control {
                border-radius: 4px;
                border: 1px solid ''' + COLOR_SCHEME['grid'] + ''';
            }
            .nav-tabs .nav-link.active {
                color: ''' + COLOR_SCHEME['primary'] + ''';
                font-weight: 500;
            }
            .nav-tabs .nav-link {
                color: ''' + COLOR_SCHEME['text'] + ''';
            }
            h1, h2, h3, h4, h5, h6 {
                color: ''' + COLOR_SCHEME['primary'] + ''';
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: ''' + COLOR_SCHEME['primary'] + ''' !important;
                color: white !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            .rc-slider-handle {
                border-color: ''' + COLOR_SCHEME['primary'] + ''';
                background-color: ''' + COLOR_SCHEME['primary'] + ''';
            }
            .rc-slider-track {
                background-color: ''' + COLOR_SCHEME['secondary'] + ''';
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define time lags for dropdown
time_lag_options = [
    {'label': 'No lag (T+0)', 'value': 'Δ Revenue_T0'},
] + [
    {'label': f'Month {i} (T+{i})', 'value': f'Δ Revenue_M{i}'} for i in range(1, 13)
] + [
    {'label': f'Year {i} (T+{i*12})', 'value': f'Δ Revenue_Y{i}'} for i in range(1, 4)
]

# Create the header with title and last updated info
header = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Interest Rate & Revenue Analytics", 
                           className="display-4 mb-0", 
                           style={"fontWeight": "600", "color": COLOR_SCHEME['primary']}),
                ])
            ], md=8),
            dbc.Col([
                html.Div([
                    html.P("Last Updated:", className="text-muted mb-0"),
                    html.Div(id="last-updated", className="h5")
                ], className="text-end")
            ], md=4),
        ])
    ]),
    className="mb-4"
)

# Create the control panel
control_panel = dbc.Card([
    dbc.CardHeader([
        html.H4("Analysis Controls", className="mb-0 d-flex align-items-center"),
        html.I(className="fas fa-sliders-h ms-2", style={"color": COLOR_SCHEME['secondary']})
    ], className="d-flex justify-content-between align-items-center"),
    dbc.CardBody([
        # Add a clearly visible toggle switch at the top of controls
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Label("Display Metrics As:", className="fw-bold me-3", 
                                 style={"color": COLOR_SCHEME['primary']}),
                        dbc.Checklist(
                            options=[
                                {"label": "Annualized", "value": "annual"},
                            ],
                            value=["annual"],  # Default to checked/annualized
                            id="metrics-display-toggle",
                            switch=True,  # Use a switch instead of checkbox
                            className="d-inline-block align-middle",
                        ),
                    ], className="d-flex align-items-center mb-0")
                ], className="mb-0")
            ], width=12, className="mb-3"),
        ]),
        # Date range filter with improved styling
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Time Range", className="fw-bold mb-2", 
                            style={"color": COLOR_SCHEME['primary']}),
                    html.Div([
                        dcc.RangeSlider(
                            id='date-range-slider',
                            min=0,
                            max=len(df) - 1,
                            value=[0, len(df) - 1],
                            marks={
                                i: {'label': df['Date'].iloc[i].strftime('%Y-%m'), 'style': {'transform': 'rotate(-45deg)', 'margin-top': '5px'}} 
                                for i in range(0, len(df), max(1, len(df) // 6))
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mt-1",
                            allowCross=False
                        ),
                    ], style={"padding": "0 10px 20px 10px"})
                ], className="p-2", 
                   style={"background": "rgba(248, 249, 250, 0.5)", "border-radius": "8px"})
            ], lg=12),
        ], className="mb-4"),
        
        # Control options with improved styling
        dbc.Row([
            dbc.Col([
                # Interest rate variable selector
                dbc.Card([
                    dbc.CardHeader("Interest Rate Variable", 
                                  className="py-2 text-white fw-medium",
                                  style={"background": COLOR_SCHEME['secondary']}),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='rate-variable-dropdown',
                            options=[
                                {'label': 'OCR', 'value': 'OCR'},
                                {'label': '30 day bill', 'value': '30 day bill'},
                                {'label': 'Δ 30 day', 'value': 'Δ 30 day'}
                            ],
                            value='Δ 30 day',
                            clearable=False,
                            className="shadow-sm"
                        ),
                    ], className="py-2")
                ], className="border-0 shadow-sm h-100")
            ], md=6),
            dbc.Col([
                # Revenue lag period selector
                dbc.Card([
                    dbc.CardHeader("Revenue Lag Period", 
                                  className="py-2 text-white fw-medium",
                                  style={"background": COLOR_SCHEME['secondary']}),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='time-lag-dropdown',
                            options=time_lag_options,
                            value='Δ Revenue_T0',  # Default to no lag
                            clearable=False,
                            className="shadow-sm"
                        ),
                    ], className="py-2")
                ], className="border-0 shadow-sm h-100")
            ], md=6),
        ], className="mb-3 g-2"),
        
        dbc.Row([
            dbc.Col([
                # Time period filter
                dbc.Card([
                    dbc.CardHeader("Time Period Filter", 
                                  className="py-2 text-white fw-medium",
                                  style={"background": COLOR_SCHEME['secondary']}),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id='time-period-radio',
                            options=[
                                {'label': 'All Data', 'value': 'all'},
                                {'label': 'Pre-COVID', 'value': 'pre_covid'},
                                {'label': 'COVID and After', 'value': 'covid'}
                            ],
                            value='all',
                            inputClassName="me-2",
                            labelClassName="me-3 fw-medium"
                        ),
                    ], className="py-2")
                ], className="border-0 shadow-sm h-100")
            ], md=6),
            dbc.Col([
                # Visualization options
                dbc.Card([
                    dbc.CardHeader("Visualization Options",
                                  className="py-2 text-white fw-medium",
                                  style={"background": COLOR_SCHEME['secondary']}),
                    dbc.CardBody([
                        dbc.Checklist(
                            options=[
                                {"label": "Moving Averages", "value": "show_ma"},
                                {"label": "Regression Line", "value": "show_reg"},
                                {"label": "Confidence Intervals", "value": "show_conf"},
                            ],
                            value=["show_reg", "show_conf"],
                            id="chart-options",
                            inline=True,
                            switch=True,
                        ),
                    ], className="py-2")
                ], className="border-0 shadow-sm h-100")
            ], md=6),
        ], className="g-2"),
    ], className="p-3"),
], className="mb-4 shadow")

# Create the main dashboard layout with a more professional look
dashboard_layout = html.Div([
    # Simplified navigation bar
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Span("Friday Musings Dashboard", className="h3 text-light font-weight-bold")
                ], width="auto"),
            ], className="g-0"),
            
            # Export button directly in the navbar
            dbc.Row([
                dbc.Col([
                    html.A(
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), "Export Data"],
                            color="light", 
                            outline=True, 
                            size="sm"
                        ),
                        id="download-link",
                        href="/download-excel",
                        download="interest_rate_analysis.xlsx",
                        target="_blank"
                    ),
                ], width="auto"),
            ], className="ms-auto")
        ], fluid=True),
        color=COLOR_SCHEME['primary'],
        dark=True,
        className="mb-4 shadow",
    ),
    
    # Main content
    dbc.Container([
        # Header and controls section
        header,
        
        dbc.Row([
            # Left sidebar with controls
            dbc.Col([
                control_panel,
                
                # Key metrics
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Key Metrics", className="mb-0"),
                        html.I(className="fas fa-chart-pie ms-2", style={"color": COLOR_SCHEME['secondary']})
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        html.Div(id="metrics-container")
                    ], className="p-0")
                ], className="mb-4 shadow"),
                
                # Correlation visualization
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Correlation Analysis", className="mb-0"),
                        html.I(className="fas fa-project-diagram ms-2", style={"color": COLOR_SCHEME['secondary']})
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        dcc.Graph(id='correlation-heatmap')
                    ], className="p-2")
                ], className="shadow"),
            ], lg=4),
            
            # Main content area
            dbc.Col([
                # Primary visualization
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Interest Rate & Revenue Time Series", className="mb-0"),
                        html.I(className="fas fa-chart-line ms-2", style={"color": COLOR_SCHEME['secondary']})
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        dcc.Graph(id='time-series-chart', style={'height': '450px'})
                    ], className="p-2")
                ], className="mb-4 shadow"),
                
                # Regression and statistical analysis
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("Regression Analysis", className="mb-0"),
                                html.I(className="fas fa-chart-scatter ms-2", style={"color": COLOR_SCHEME['secondary']})
                            ], className="d-flex justify-content-between align-items-center"),
                            dbc.CardBody([
                                dcc.Graph(id='correlation-chart', style={'height': '350px'})
                            ], className="p-2")
                        ], className="h-100 shadow")
                    ], lg=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("Statistical Analysis", className="mb-0"),
                                html.I(className="fas fa-calculator ms-2", style={"color": COLOR_SCHEME['secondary']})
                            ], className="d-flex justify-content-between align-items-center"),
                            dbc.CardBody([
                                html.Div(id='stats-table-container', style={'height': '350px', 'overflow': 'auto'})
                            ], className="p-3")
                        ], className="h-100 shadow")
                    ], lg=6),
                ], className="mb-4 g-4"),
                
                # Advanced analysis tabs
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Advanced Analytics", className="mb-0"),
                        html.I(className="fas fa-brain ms-2", style={"color": COLOR_SCHEME['secondary']})
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(dcc.Graph(id='distribution-chart'), 
                                   label="Distributions", 
                                   tab_id="tab-dist",
                                   label_style={"font-weight": "bold", "color": COLOR_SCHEME['primary']}),
                            dbc.Tab(dcc.Graph(id='seasonal-decomposition'), 
                                   label="Seasonal Analysis", 
                                   tab_id="tab-seasonal",
                                   label_style={"font-weight": "bold", "color": COLOR_SCHEME['primary']}),
                            dbc.Tab(html.Div(id='regression-output', className="p-3"), 
                                   label="Detailed Statistics", 
                                   tab_id="tab-stats",
                                   label_style={"font-weight": "bold", "color": COLOR_SCHEME['primary']})
                        ], id="tabs", active_tab="tab-dist")
                    ], className="p-2")
                ], className="mb-4 shadow"),
                
                # Data table
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Data Explorer", className="mb-0"),
                        html.I(className="fas fa-table ms-2", style={"color": COLOR_SCHEME['secondary']})
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        html.Div(id='table-container')
                    ], className="p-0")
                ], className="shadow"),
            ], lg=8),
        ]),
    ], fluid=True, className="mt-0 mb-5"),
    
    # Footer
    html.Footer([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.P("Interest Rate & Revenue Analytics Dashboard", className="mb-0 text-white")
                ], md=6),
                dbc.Col([
                    html.P("© 2025 Financial Analytics Ltd.", className="mb-0 text-white text-end")
                ], md=6),
            ])
        ], fluid=True)
    ], className="bg-dark text-white py-3 mt-4")
])

# Set the app layout
app.layout = dashboard_layout

# Callback to convert date slider to actual dates
def get_date_from_index(index):
    if isinstance(index, list):
        return [df['Date'].iloc[i] for i in index]
    return df['Date'].iloc[index]

# Callback for key metrics cards
@app.callback(
    Output('metrics-container', 'children'),
    [Input('date-range-slider', 'value'),
     Input('time-period-radio', 'value'),
     Input('rate-variable-dropdown', 'value'),
     Input('time-lag-dropdown', 'value'),
     Input('metrics-display-toggle', 'value')]  # Now using checklist value (list)
)
def update_metrics(date_range_indices, time_period, rate_variable, time_lag_column, display_mode_list):
    # Convert slider indices to actual dates
    date_range = get_date_from_index(date_range_indices)
    
    # Filter the data based on the selected date range
    filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    
    # Apply additional time period filter if selected
    if time_period == 'pre_covid':
        filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
    elif time_period == 'covid':
        filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
    
    # Calculate metrics
    try:
        # Determine display mode from checklist (which returns a list of selected values)
        # If 'annual' is in the list, use annualized mode
        display_mode = 'annual' if 'annual' in display_mode_list else 'monthly'
        
        # Calculate base values (monthly)
        avg_revenue_monthly = filtered_df['Total'].mean()
        revenue_growth_monthly = filtered_df['Δ Revenue'].mean()
        avg_rate_monthly = filtered_df[rate_variable].mean()
        
        # Convert to annualized values if selected
        if display_mode == 'annual':
            # Annualize the metrics
            avg_revenue = avg_revenue_monthly * 12  # Annual total
            
            # Convert to millions for cleaner display
            avg_revenue_in_millions = avg_revenue / 1000000
            
            # Convert to annual compounded growth rate
            revenue_growth = ((1 + revenue_growth_monthly) ** 12 - 1) * 100
            
            # For rates, we can either keep monthly (as most rates are quoted) 
            # or convert to annual equivalent rate
            # Here we'll keep the rate as is but label as annual equivalent
            avg_rate = avg_rate_monthly * 100
            
            # Add label indicators for annualized display
            revenue_label = "Annual Revenue"
            revenue_display = f"${avg_revenue_in_millions:.1f}M"  # Display in millions
            growth_label = "Annual Growth"
            rate_label = f"Annual {rate_variable}"
        else:
            # Use monthly values
            avg_revenue = avg_revenue_monthly
            revenue_growth = revenue_growth_monthly * 100  # Convert to percentage
            avg_rate = avg_rate_monthly * 100
            
            # Labels for monthly display
            revenue_label = "Monthly Revenue"
            revenue_display = f"${avg_revenue:,.0f}"
            growth_label = "Monthly Growth"
            rate_label = f"Monthly {rate_variable}"
        
        # Calculate correlation based on selected lag period
        # The correlation is not affected by the annualized/monthly toggle
        # Only include rows with valid data
        corr_df = filtered_df.dropna(subset=[rate_variable, time_lag_column])
        if len(corr_df) > 3:  # Need at least 3 points for correlation
            correlation = corr_df[rate_variable].corr(corr_df[time_lag_column])
        else:
            correlation = np.nan
        
        # Define color based on correlation strength
        if np.isnan(correlation):
            corr_color = "secondary"
        elif abs(correlation) < 0.3:
            corr_color = "warning"
        elif correlation >= 0.3:
            corr_color = "success"
        else:
            corr_color = "danger"
        
        # Text color - use the same for all cards
        text_color = COLOR_SCHEME['primary']
        
        # Create the metrics cards with fixed styling
        metrics_cards = dbc.Row([
            # Revenue Card
            dbc.Col(dbc.Card([
                dbc.CardHeader(revenue_label, 
                               className="py-2", 
                               style={"backgroundColor": "#f8f9fa", "color": text_color, "fontWeight": "500"}),
                dbc.CardBody([
                    html.H2(revenue_display, 
                           style={"color": text_color, "fontWeight": "600", "textAlign": "center", "marginBottom": "0"})
                ], className="py-3")
            ], className="border"), width=3),
            
            # Revenue Growth Card
            dbc.Col(dbc.Card([
                dbc.CardHeader(growth_label, 
                               className="py-2", 
                               style={"backgroundColor": "#f8f9fa", "color": text_color, "fontWeight": "500"}),
                dbc.CardBody([
                    html.H2(
                        f"{revenue_growth:.2f}%", 
                        style={
                            "color": "#27AE60" if revenue_growth > 0 else "#E74C3C", 
                            "fontWeight": "600", 
                            "textAlign": "center",
                            "marginBottom": "0"
                        }
                    ),
                ], className="py-3")
            ], className="border"), width=3),
            
            # Rate Card
            dbc.Col(dbc.Card([
                dbc.CardHeader(rate_label, 
                               className="py-2", 
                               style={"backgroundColor": "#f8f9fa", "color": text_color, "fontWeight": "500"}),
                dbc.CardBody([
                    html.H2(f"{avg_rate:.2f}%", 
                           style={"color": text_color, "fontWeight": "600", "textAlign": "center", "marginBottom": "0"})
                ], className="py-3")
            ], className="border"), width=3),
            
            # Correlation Card
            dbc.Col(dbc.Card([
                dbc.CardHeader("Correlation", 
                               className="py-2", 
                               style={"backgroundColor": "#f8f9fa", "color": text_color, "fontWeight": "500"}),
                dbc.CardBody([
                    html.H2(
                        f"{correlation:.3f}" if not np.isnan(correlation) else "Insufficient data", 
                        style={
                            "color": text_color, 
                            "fontWeight": "600", 
                            "textAlign": "center",
                            "marginBottom": "0"
                        }
                    ),
                ], className="py-3")
            ], className="border"), width=3),
        ], className="gy-0")
        
        return metrics_cards
    
    except Exception as e:
        print(f"Error updating metrics: {e}")
        return html.Div(f"Error calculating metrics: {str(e)}")

# Callback for the correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('date-range-slider', 'value'),
     Input('time-period-radio', 'value'),
     Input('rate-variable-dropdown', 'value')]
)
def update_correlation_heatmap(date_range_indices, time_period, rate_variable):
    try:
        # Convert slider indices to actual dates
        date_range = get_date_from_index(date_range_indices)
        
        # Filter the data based on the selected date range
        filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
        
        # Apply additional time period filter if selected
        if time_period == 'pre_covid':
            filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
        elif time_period == 'covid':
            filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
        
        # Create a better visualization for correlation analysis
        # Instead of heatmap with 2 rows, create a single heatmap with all lags
        
        # Prepare labels and correlation values for all time periods
        labels = ['T+0', 'T+1m', 'T+2m', 'T+3m', 'T+4m', 'T+5m', 'T+6m', 
                 'T+7m', 'T+8m', 'T+9m', 'T+10m', 'T+11m', 'T+1yr', 'T+2yr', 'T+3yr']
        
        # Column names for monthly and yearly lags
        columns = ['Δ Revenue_T0'] + [f'Δ Revenue_M{i}' for i in range(1, 12)] + ['Δ Revenue_Y1', 'Δ Revenue_Y2', 'Δ Revenue_Y3']
        
        # Calculate correlations
        corr_values = []
        for col in columns:
            if col in filtered_df.columns:
                valid_data = filtered_df.dropna(subset=[rate_variable, col])
                if len(valid_data) > 2:  # Need at least 3 points for correlation
                    corr = valid_data[rate_variable].corr(valid_data[col])
                    corr_values.append(corr)
                else:
                    corr_values.append(np.nan)
            else:
                print(f"Column {col} not found in dataframe")
                corr_values.append(np.nan)
        
        # Find min/max correlation for better color scaling
        valid_corrs = [c for c in corr_values if not np.isnan(c)]
        
        # Set default min/max if no valid correlations
        if not valid_corrs:
            valid_corrs = [-1, 1]
        
        # Create figure for bar chart visualization
        fig = go.Figure()
        
        # Add bars for correlations
        bar_colors = [
            COLOR_SCHEME['correlation_negative'] if c < 0 else COLOR_SCHEME['correlation_positive'] 
            for c in corr_values
        ]
        
        # Increase intensity for stronger correlations
        bar_colors = [
            f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, {min(0.3 + 0.7 * abs(v), 1) if not np.isnan(v) else 0.1})"
            for c, v in zip(bar_colors, corr_values)
        ]
        
        # Add correlation bars with complete redesign
        # Use numeric x-axis for better positioning and remove text to prevent duplicates
        fig.add_trace(go.Bar(
            x=list(range(len(labels))),  # Use numeric indices instead of labels
            y=corr_values,
            marker_color=bar_colors,
            marker_line_width=1,
            marker_line_color='rgba(0,0,0,0.1)',
            hovertemplate='%{text}: %{y:.3f}<extra></extra>',
            text=labels,  # Use labels in hover text
            textposition='none',  # Don't show text on bars
            width=0.7,  # Width of bars (max 1.0)
        ))
        
        # Add text labels on top of or below bars
        for i, (corr, label) in enumerate(zip(corr_values, labels)):
            if not np.isnan(corr):
                fig.add_annotation(
                    x=i,
                    y=corr + (0.05 if corr >= 0 else -0.05),
                    text=f"{corr:.2f}",
                    showarrow=False,
                    font=dict(
                        color=COLOR_SCHEME['text'],
                        size=10,
                        family="Arial"
                    ),
                    yanchor="bottom" if corr >= 0 else "top"
                )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(labels) - 0.5,
            y0=0,
            y1=0,
            line=dict(color=COLOR_SCHEME['grid'], width=1, dash="dot"),
        )
        
        # Find the maximum absolute correlation
        if valid_corrs:
            max_abs_idx = np.nanargmax(np.abs(corr_values))
            # Create better highlight for maximum correlation value
            # Add a colored box highlight
            fig.add_trace(go.Bar(
                x=[max_abs_idx],  # Position at the max correlation
                y=[corr_values[max_abs_idx]],
                marker_color='rgba(0,0,0,0)',  # Transparent fill
                marker_line_width=3,  # Thicker border
                marker_line_color=COLOR_SCHEME['highlight'],  # Highlight color
                width=0.85,  # Slightly wider than regular bars
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add "Strongest" label
            fig.add_annotation(
                x=max_abs_idx,
                y=corr_values[max_abs_idx] + (0.15 if corr_values[max_abs_idx] >= 0 else -0.15),
                text="Strongest",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=COLOR_SCHEME['highlight'],
                font=dict(
                    color=COLOR_SCHEME['highlight'],
                    size=10,
                    family="Arial"
                ),
                yanchor="bottom" if corr_values[max_abs_idx] >= 0 else "top"
            )
        
        # Remove annotations and put text in the bars instead
        # The issue was duplicate text from annotations and text property in bars
        
        # Enhanced styling with completely new approach
        fig.update_layout(
            title=dict(
                text=f"Correlation between {rate_variable} and Revenue Change by Time Lag",
                font=dict(size=16, color=COLOR_SCHEME['primary'])
            ),
            xaxis=dict(
                title="Time Lag Period",
                tickmode='array',
                tickvals=list(range(len(labels))),
                ticktext=labels,
                tickangle=-30,
                tickfont=dict(size=11),
                # Numeric axis for better positioning
                type='linear',
                fixedrange=False  # Allow zooming if needed
            ),
            yaxis=dict(
                title="Correlation Coefficient",
                range=[-1.05, 1.05],
                zeroline=False,
                gridcolor=COLOR_SCHEME['grid'],
                # Add minor gridlines
                minor=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor=COLOR_SCHEME['grid']
                )
            ),
            plot_bgcolor=COLOR_SCHEME['widget_bg'],
            paper_bgcolor=COLOR_SCHEME['widget_bg'],
            margin=dict(l=40, r=20, t=60, b=60),
            height=350,
            hovermode="x unified",
            # Additional settings for cleaner rendering
            barmode='relative',
            bargap=0.15,
            uniformtext=dict(
                mode='hide',
                minsize=10
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating correlation heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLOR_SCHEME['negative'])
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SCHEME['widget_bg'],
            paper_bgcolor=COLOR_SCHEME['widget_bg'],
            height=350
        )
        return fig

# Callback for the statistical analysis table
@app.callback(
    Output('stats-table-container', 'children'),
    [Input('date-range-slider', 'value'),
     Input('time-period-radio', 'value'),
     Input('rate-variable-dropdown', 'value'),
     Input('time-lag-dropdown', 'value')]
)
def update_stats_table(date_range_indices, time_period, rate_variable, time_lag_column):
    try:
        # Convert slider indices to actual dates
        date_range = get_date_from_index(date_range_indices)
        
        # Filter the data based on the selected date range
        filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
        
        # Apply additional time period filter if selected
        if time_period == 'pre_covid':
            filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
        elif time_period == 'covid':
            filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
        
        # Get valid data for regression
        valid_data = filtered_df.dropna(subset=[rate_variable, time_lag_column])
        
        if len(valid_data) < 5:
            return html.Div("Insufficient data for statistical analysis (need at least 5 data points)",
                           className="text-danger p-4")
        
        # Perform regression analysis
        X = valid_data[rate_variable].values.reshape(-1, 1)
        y = valid_data[time_lag_column].values
        
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        # Extract all key statistics
        stats = {
            'Observations': len(valid_data),
            'R-squared': model.rsquared,
            'Adjusted R-squared': model.rsquared_adj,
            'F-statistic': model.fvalue,
            'Prob (F-statistic)': model.f_pvalue,
            'Log-Likelihood': model.llf,
            'AIC': model.aic,
            'BIC': model.bic,
            'Pearson Correlation': valid_data[rate_variable].corr(valid_data[time_lag_column]),
            'Durbin-Watson': sm.stats.stattools.durbin_watson(model.resid),
            'Jarque-Bera (p-value)': sm.stats.jarque_bera(model.resid)[1],
            'Skew of Residuals': pd.Series(model.resid).skew(),
            'Kurtosis of Residuals': pd.Series(model.resid).kurtosis()
        }
        
        # Create a table of coefficient values
        coef_stats = model.summary2().tables[1]
        coef_data = []
        for i, row in enumerate(coef_stats.index):
            coef_data.append({
                'Parameter': 'Intercept' if i == 0 else rate_variable,
                'Coefficient': coef_stats.loc[row, 'Coef.'],
                'Std Error': coef_stats.loc[row, 'Std.Err.'],
                't-value': coef_stats.loc[row, 't'],
                'p-value': coef_stats.loc[row, 'P>|t|'],
                'Lower 95% CI': coef_stats.loc[row, '[0.025'],
                'Upper 95% CI': coef_stats.loc[row, '0.975]']
            })
        
        # Create the general statistics table
        general_stats_table = dash_table.DataTable(
            id='general-stats-table',
            columns=[
                {'name': 'Statistic', 'id': 'Statistic'},
                {'name': 'Value', 'id': 'Value'}
            ],
            data=[{'Statistic': k, 'Value': f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in stats.items()],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': COLOR_SCHEME['primary'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
        )
        
        # Create the coefficient table
        coef_table = dash_table.DataTable(
            id='coef-stats-table',
            columns=[{'name': k, 'id': k} for k in coef_data[0].keys()],
            data=coef_data,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': COLOR_SCHEME['primary'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'filter_query': '{p-value} < 0.05'},
                    'backgroundColor': 'rgba(76, 175, 80, 0.2)',
                    'fontWeight': 'bold'
                }
            ],
        )
        
        # Return the tables with headers
        return html.Div([
            html.H5("Regression Model Statistics"),
            general_stats_table,
            html.H5("Coefficient Statistics", className="mt-4"),
            coef_table,
            html.Div([
                html.P("Statistical interpretation:", className="fw-bold mt-3"),
                html.Ul([
                    html.Li(f"The model explains {stats['R-squared']:.1%} of the variance in revenue change."),
                    html.Li(f"A 1% change in {rate_variable} is associated with a {abs(coef_data[1]['Coefficient']):.2%} "
                           f"{'increase' if coef_data[1]['Coefficient'] > 0 else 'decrease'} in revenue."),
                    html.Li(f"This relationship is {'statistically significant' if coef_data[1]['p-value'] < 0.05 else 'not statistically significant'} "
                           f"(p = {coef_data[1]['p-value']:.4f})."),
                ])
            ])
        ])
    except Exception as e:
        print(f"Error creating stats table: {e}")
        return html.Div(f"Error creating statistical analysis: {str(e)}", className="text-danger p-4")

# Create the main callbacks for visualizations
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('correlation-chart', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('seasonal-decomposition', 'figure'),
     Output('regression-output', 'children'),
     Output('table-container', 'children'),
     Output('last-updated', 'children')],
    [Input('date-range-slider', 'value'),
     Input('time-period-radio', 'value'),
     Input('rate-variable-dropdown', 'value'),
     Input('time-lag-dropdown', 'value'),
     Input('chart-options', 'value')]
)
def update_charts(date_range_indices, time_period, rate_variable, time_lag_column, chart_options):
    # Get current time for last updated
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_updated = f"Last updated: {current_time}"
    
    # Convert slider indices to actual dates
    date_range = get_date_from_index(date_range_indices)
    
    # Filter the data based on the selected date range
    filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    
    # Apply additional time period filter if selected
    if time_period == 'pre_covid':
        filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
    elif time_period == 'covid':
        filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
    
    # Create the time series chart with enhanced styling
    try:
        fig_time_series = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue to the primary y-axis with enhanced styling
        fig_time_series.add_trace(
            go.Scatter(
                x=filtered_df['Date'], 
                y=filtered_df['Total'], 
                name='Revenue',
                line=dict(color=COLOR_SCHEME['revenue'], width=2),
                hovertemplate='Date: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add selected interest rate to the secondary y-axis with enhanced styling
        fig_time_series.add_trace(
            go.Scatter(
                x=filtered_df['Date'], 
                y=filtered_df[rate_variable], 
                name=rate_variable,
                line=dict(color=COLOR_SCHEME['interest'], width=2),
                hovertemplate='Date: %{x}<br>' + rate_variable + ': %{y:.2%}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add moving average if selected
        if "show_ma" in chart_options:
            # Add 3-month moving average
            fig_time_series.add_trace(
                go.Scatter(
                    x=filtered_df['Date'], 
                    y=filtered_df['Revenue_MA_3M'], 
                    name='Revenue 3M MA',
                    line=dict(color=COLOR_SCHEME['revenue'], width=1.5, dash='dot'),
                    hovertemplate='Date: %{x}<br>3M Avg: $%{y:,.2f}<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add 12-month moving average
            fig_time_series.add_trace(
                go.Scatter(
                    x=filtered_df['Date'], 
                    y=filtered_df['Revenue_MA_12M'], 
                    name='Revenue 12M MA',
                    line=dict(color=COLOR_SCHEME['revenue'], width=2, dash='dash'),
                    hovertemplate='Date: %{x}<br>12M Avg: $%{y:,.2f}<extra></extra>'
                ),
                secondary_y=False
            )
        
        # Add annotations for important events
        if COVID_START >= filtered_df['Date'].min() and COVID_START <= filtered_df['Date'].max():
            # Add vertical line for COVID start
            fig_time_series.add_shape(
                type="line",
                x0=COVID_START,
                x1=COVID_START,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add annotation for COVID
            fig_time_series.add_annotation(
                x=COVID_START,
                y=1,
                yref="paper",
                text="COVID-19",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                arrowsize=1,
                arrowwidth=2,
            )
        
        # Enhanced layout styling
        fig_time_series.update_layout(
            title=dict(
                text="Revenue and Interest Rate Over Time",
                font=dict(size=20, color=COLOR_SCHEME['primary'])
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLOR_SCHEME['grid']
            ),
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            margin=dict(l=20, r=20, t=60, b=20),
        )
        
        fig_time_series.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor=COLOR_SCHEME['grid'],
            zeroline=False,
            showline=True,
            linecolor=COLOR_SCHEME['primary'],
            tickfont=dict(color=COLOR_SCHEME['text'])
        )
        
        fig_time_series.update_yaxes(
            title_text="Revenue ($)",
            showgrid=True,
            gridcolor=COLOR_SCHEME['grid'],
            zeroline=False,
            showline=True,
            linecolor=COLOR_SCHEME['primary'],
            tickformat="$,.0f",
            secondary_y=False,
            tickfont=dict(color=COLOR_SCHEME['revenue'])
        )
        
        fig_time_series.update_yaxes(
            title_text=rate_variable,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=COLOR_SCHEME['primary'],
            tickformat=".2%",
            secondary_y=True,
            tickfont=dict(color=COLOR_SCHEME['interest'])
        )
    except Exception as e:
        print(f"Error creating time series chart: {e}")
        # Create empty figure with error message
        fig_time_series = go.Figure()
        fig_time_series.add_annotation(
            text=f"Error creating time series chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    
    # Create the correlation scatter plot with enhanced styling
    try:
        valid_scatter_data = filtered_df.dropna(subset=[rate_variable, time_lag_column])
        
        # Only try to add trendline if we have sufficient data
        use_trendline = "show_reg" in chart_options and len(valid_scatter_data) >= 3
        
        # Extract display name for the time lag
        lag_display_name = next((opt['label'] for opt in time_lag_options if opt['value'] == time_lag_column), time_lag_column)
        
        fig_correlation = px.scatter(
            valid_scatter_data, 
            x=rate_variable, 
            y=time_lag_column,
            trendline='ols' if use_trendline else None,
            labels={
                rate_variable: rate_variable,
                time_lag_column: "Revenue Change"
            },
            title=f"Correlation between {rate_variable} and Revenue Change ({lag_display_name})",
            template="plotly_white",
            color_discrete_sequence=[COLOR_SCHEME['accent']]
        )
        
        # Add confidence intervals if selected
        if "show_conf" in chart_options and use_trendline and len(valid_scatter_data) >= 5:
            # Perform regression to get confidence intervals
            X = sm.add_constant(valid_scatter_data[rate_variable])
            y = valid_scatter_data[time_lag_column]
            model = sm.OLS(y, X).fit()
            
            # Generate predictions with confidence intervals
            x_range = np.linspace(valid_scatter_data[rate_variable].min(), valid_scatter_data[rate_variable].max(), 100)
            X_pred = sm.add_constant(x_range)
            y_pred = model.predict(X_pred)
            
            # Get prediction confidence intervals
            ci = model.get_prediction(X_pred).conf_int(0.95)
            
            # Add confidence interval band
            fig_correlation.add_trace(
                go.Scatter(
                    x=np.concatenate([x_range, x_range[::-1]]),
                    y=np.concatenate([ci[:, 0], ci[::-1, 1]]),
                    fill='toself',
                    fillcolor='rgba(200, 200, 200, 0.3)',
                    line=dict(color='rgba(0, 0, 0, 0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    name='95% Confidence Interval'
                )
            )
        
        # Enhanced layout styling
        fig_correlation.update_layout(
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            margin=dict(l=20, r=20, t=60, b=20),
            title=dict(
                font=dict(size=16, color=COLOR_SCHEME['primary'])
            ),
            xaxis=dict(
                title=dict(font=dict(size=14)),
                showgrid=True,
                gridcolor=COLOR_SCHEME['grid'],
                zeroline=False,
                tickformat=".2%"
            ),
            yaxis=dict(
                title=dict(font=dict(size=14)),
                showgrid=True,
                gridcolor=COLOR_SCHEME['grid'],
                zeroline=False,
                tickformat=".2%"
            )
        )
    except Exception as e:
        print(f"Error creating correlation chart: {e}")
        # Create empty figure with error message
        fig_correlation = go.Figure()
        fig_correlation.add_annotation(
            text=f"Error creating correlation chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    
    # Create the distribution chart with enhanced styling
    try:
        fig_distribution = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15)
        
        # Add histogram for rate variable
        valid_rate_data = filtered_df.dropna(subset=[rate_variable])
        if len(valid_rate_data) > 0:
            # Add histogram
            fig_distribution.add_trace(
                go.Histogram(
                    x=valid_rate_data[rate_variable],
                    name=rate_variable,
                    opacity=0.7,
                    marker=dict(
                        color=COLOR_SCHEME['interest'],
                        line=dict(color='white', width=0.5)
                    ),
                    autobinx=True,
                    hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add histogram for revenue change
        if time_lag_column in filtered_df.columns:
            valid_revenue_data = filtered_df.dropna(subset=[time_lag_column])
            if len(valid_revenue_data) > 0:
                # Add histogram
                fig_distribution.add_trace(
                    go.Histogram(
                        x=valid_revenue_data[time_lag_column],
                        name="Revenue Change",
                        opacity=0.7,
                        marker=dict(
                            color=COLOR_SCHEME['revenue'],
                            line=dict(color='white', width=0.5)
                        ),
                        autobinx=True,
                        hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Enhanced layout styling
        fig_distribution.update_layout(
            title=dict(
                text="Distribution of Variables",
                font=dict(size=16, color=COLOR_SCHEME['primary'])
            ),
            showlegend=False,
            height=400,
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            margin=dict(l=20, r=20, t=60, b=20),
        )
        
        # Update axes
        fig_distribution.update_xaxes(
            title_text=rate_variable,
            row=1, col=1,
            showgrid=True,
            gridcolor=COLOR_SCHEME['grid'],
            tickformat=".2%"
        )
        
        lag_display_name = next((opt['label'] for opt in time_lag_options if opt['value'] == time_lag_column), "Revenue Change")
        fig_distribution.update_xaxes(
            title_text=f"Revenue Change ({lag_display_name})",
            row=2, col=1,
            showgrid=True,
            gridcolor=COLOR_SCHEME['grid'],
            tickformat=".2%"
        )
        
        fig_distribution.update_yaxes(
            title_text="Frequency",
            showgrid=True,
            gridcolor=COLOR_SCHEME['grid']
        )
    except Exception as e:
        print(f"Error creating distribution chart: {e}")
        # Create empty figure with error message
        fig_distribution = go.Figure()
        fig_distribution.add_annotation(
            text=f"Error creating distribution chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    
    # Create seasonal decomposition chart
    try:
        if len(filtered_df) >= 24:  # Need at least 2 years of data
            # Use revenue for seasonal decomposition
            ts_data = filtered_df.set_index('Date')['Total']
            
            # Perform seasonal decomposition
            result = seasonal_decompose(ts_data, model='multiplicative', period=12)
            
            # Create the chart
            fig_seasonal = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
            )
            
            # Add traces
            fig_seasonal.add_trace(
                go.Scatter(
                    x=result.observed.index, y=result.observed,
                    mode='lines', name='Observed',
                    line=dict(color=COLOR_SCHEME['primary'], width=1.5)
                ),
                row=1, col=1
            )
            
            fig_seasonal.add_trace(
                go.Scatter(
                    x=result.trend.index, y=result.trend,
                    mode='lines', name='Trend',
                    line=dict(color=COLOR_SCHEME['secondary'], width=1.5)
                ),
                row=2, col=1
            )
            
            fig_seasonal.add_trace(
                go.Scatter(
                    x=result.seasonal.index, y=result.seasonal,
                    mode='lines', name='Seasonal',
                    line=dict(color=COLOR_SCHEME['accent'], width=1.5)
                ),
                row=3, col=1
            )
            
            fig_seasonal.add_trace(
                go.Scatter(
                    x=result.resid.index, y=result.resid,
                    mode='lines', name='Residual',
                    line=dict(color=COLOR_SCHEME['neutral'], width=1)
                ),
                row=4, col=1
            )
            
            # Update layout
            fig_seasonal.update_layout(
                height=600,
                showlegend=False,
                title=dict(
                    text="Seasonal Decomposition of Revenue",
                    font=dict(size=16, color=COLOR_SCHEME['primary'])
                ),
                plot_bgcolor=COLOR_SCHEME['background'],
                paper_bgcolor=COLOR_SCHEME['background'],
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # Update y-axes titles
            fig_seasonal.update_yaxes(title_text="Revenue", row=1, col=1)
            fig_seasonal.update_yaxes(title_text="Trend", row=2, col=1)
            fig_seasonal.update_yaxes(title_text="Seasonal Factor", row=3, col=1)
            fig_seasonal.update_yaxes(title_text="Residual", row=4, col=1)
        else:
            # Not enough data for seasonal decomposition
            fig_seasonal = go.Figure()
            fig_seasonal.add_annotation(
                text="Insufficient data for seasonal decomposition (need at least 24 months)",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
    except Exception as e:
        print(f"Error creating seasonal decomposition: {e}")
        # Create empty figure with error message
        fig_seasonal = go.Figure()
        fig_seasonal.add_annotation(
            text=f"Error creating seasonal decomposition: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    
    # Create detailed regression output
    try:
        # Check if we have enough data
        valid_data = filtered_df.dropna(subset=[rate_variable, time_lag_column])
        
        if len(valid_data) < 5:
            regression_output = html.Div([
                html.H5("Insufficient Data", className="text-warning"),
                html.P("Not enough data points for detailed regression analysis (need at least 5 points).")
            ])
        else:
            # Perform regression
            X = valid_data[rate_variable].values.reshape(-1, 1)
            y = valid_data[time_lag_column].values
            
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm).fit()
            
            # Convert the summary to HTML
            summary_html = model.summary().tables[0].as_html() + model.summary().tables[1].as_html() + model.summary().tables[2].as_html()
            
            # Create the regression output with enhanced styling
            regression_output = html.Div([
                html.H5("Detailed Regression Results"),
                html.Div([
                    html.Iframe(
                        srcDoc=summary_html,
                        style={'width': '100%', 'height': '500px', 'border': 'none'}
                    )
                ]),
                html.Hr(),
                html.H5("Interpretation"),
                html.P([
                    f"The regression model shows that {rate_variable} ", 
                    html.B(f"{'positively' if model.params[1] > 0 else 'negatively'} affects"),
                    f" revenue change with lag {lag_display_name}. ",
                    f"A 1% {'increase' if model.params[1] > 0 else 'decrease'} in {rate_variable} is associated with a {abs(model.params[1]):.2%} ",
                    f"{'increase' if model.params[1] > 0 else 'decrease'} in revenue change."
                ]),
                html.P([
                    f"This relationship is ", 
                    html.B(f"{'statistically significant' if model.f_pvalue < 0.05 else 'not statistically significant'}"),
                    f" (p = {model.f_pvalue:.4f})."
                ]),
                html.P([
                    f"The model explains ", 
                    html.B(f"{model.rsquared:.1%}"),
                    f" of the variance in revenue change."
                ])
            ])
    except Exception as e:
        print(f"Error in regression analysis: {e}")
        regression_output = html.Div([
            html.H5("Regression Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ])
    
    # Prepare data for the table with enhanced styling
    try:
        table_df = filtered_df.copy()  # Don't drop NAs to show all data
        
        # Determine which columns to include
        table_columns = ['Date', 'Total', rate_variable, 'Δ Revenue', time_lag_column]
        
        # Only include columns that exist in the dataframe
        table_columns = [col for col in table_columns if col in table_df.columns]
        
        # Generate table data
        table_data = table_df[table_columns].to_dict('records')
        
        # Create formatted columns for the table
        formatted_columns = []
        for col in table_columns:
            if col == 'Date':
                formatted_columns.append({"name": "Date", "id": "Date", "type": "datetime"})
            elif col in ['Δ Revenue', 'Δ 30 day', 'OCR', '30 day bill'] or col.startswith('Δ Revenue_'):
                formatted_columns.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ".2%"}})
            else:
                formatted_columns.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ",.2f"}})
        
        # Create the table component
        table_component = dash_table.DataTable(
            id='data-table',
            data=table_data,
            columns=formatted_columns,
            sort_action='native',
            filter_action='native',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': COLOR_SCHEME['primary'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'filter_query': '{Δ Revenue} > 0'},
                    'backgroundColor': 'rgba(76, 175, 80, 0.2)'
                },
                {
                    'if': {'filter_query': '{Δ Revenue} < 0'},
                    'backgroundColor': 'rgba(244, 67, 54, 0.2)'
                }
            ],
        )
    except Exception as e:
        print(f"Error preparing data table: {e}")
        # Provide fallback empty table
        table_component = html.Div([
            html.P(f"Error creating table: {str(e)}", className="text-danger")
        ])
    
    return fig_time_series, fig_correlation, fig_distribution, fig_seasonal, regression_output, table_component, last_updated

# Add a server route for downloading the data as Excel
@app.server.route("/download-excel")
def download_excel():
    from flask import send_file
    import io
    import pandas as pd
    from datetime import datetime
    
    # Create a filtered version of the dataset
    output = io.BytesIO()
    
    # Create Excel writer
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Write current filtered data to Excel
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    
    # Create analysis sheet
    analysis_df = pd.DataFrame({
        'Metric': ['Average Revenue', 'Average Revenue Growth', 'Average OCR', 'Average 30-day Bill'],
        'Value': [
            f"${df['Total'].mean():,.2f}",
            f"{df['Δ Revenue'].mean()*100:.2f}%",
            f"{df['OCR'].mean()*100:.2f}%",
            f"{df['30 day bill'].mean()*100:.2f}%"
        ]
    })
    
    # Write analysis to Excel
    analysis_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Calculate correlations for different lags
    corr_data = []
    for lag in range(0, 13):
        lag_col = 'Δ Revenue_T0' if lag == 0 else f'Δ Revenue_M{lag}'
        if lag_col in df.columns:
            # Get correlation between interest rate and revenue with this lag
            valid_data = df.dropna(subset=['Δ 30 day', lag_col])
            if len(valid_data) > 3:
                corr = valid_data['Δ 30 day'].corr(valid_data[lag_col])
                label = f"No Lag" if lag == 0 else f"{lag} {'Month' if lag == 1 else 'Months'}"
                corr_data.append({'Lag': label, 'Correlation': corr})
    
    # Add yearly lags
    for year in range(1, 4):
        lag_col = f'Δ Revenue_Y{year}'
        if lag_col in df.columns:
            valid_data = df.dropna(subset=['Δ 30 day', lag_col])
            if len(valid_data) > 3:
                corr = valid_data['Δ 30 day'].corr(valid_data[lag_col])
                corr_data.append({'Lag': f"{year} {'Year' if year == 1 else 'Years'}", 'Correlation': corr})
    
    # Create correlation DataFrame
    corr_df = pd.DataFrame(corr_data)
    
    # Write correlation data to Excel
    if not corr_df.empty:
        corr_df.to_excel(writer, sheet_name='Correlations', index=False)
    
    # Add metadata sheet
    meta_df = pd.DataFrame({
        'Property': ['Generated On', 'Data Range', 'Number of Observations', 'File Source'],
        'Value': [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
            str(len(df)),
            'Friday Musings Dashboard'
        ]
    })
    
    # Write metadata to Excel
    meta_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    # Save Excel file
    writer.close()
    
    # Set pointer to beginning of stream
    output.seek(0)
    
    # Generate filename with current date
    filename = f"Interest_Rate_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
    
    # Return Excel file as download
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    import os
    
    # Use port assigned by the hosting platform if available, or default to 8050
    port = int(os.environ.get("PORT", 8050))
    
    # Check if running in development or production
    is_dev = not os.environ.get("RENDER", False)
    
    # Run server
    app.run_server(
        host="0.0.0.0",  # Necessary for production deployment
        port=port,
        debug=is_dev  # Only enable debug in development
    )