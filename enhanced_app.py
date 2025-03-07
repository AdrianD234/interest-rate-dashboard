import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
try:
    import dash_daq as daq
    import dash_mantine_components as dmc
    from dash_extensions import Download
    from dash_extensions.enrich import DashProxy, ServersideOutputTransform, MultiplexerTransform
    import dash_ag_grid as dag
    HAS_ADVANCED_COMPONENTS = True
except ImportError:
    HAS_ADVANCED_COMPONENTS = False
    print("Some advanced components couldn't be imported. Using basic components instead.")

from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import warnings

# Try to import advanced analysis libraries
try:
    import ppscore as pps  # Predictive Power Score
    import pmdarima as pm  # Auto ARIMA for forecasting
    HAS_ADVANCED_ANALYSIS = True
except ImportError:
    HAS_ADVANCED_ANALYSIS = False
    print("Advanced analysis libraries not available. Using basic analysis methods.")

warnings.filterwarnings('ignore')

# Set Plotly template to a modern, visually appealing style
pio.templates.default = "plotly_white"

# Define custom color scheme
COLOR_SCHEME = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'accent': '#E74C3C',
    'background': '#ECF0F1',
    'text': '#2C3E50',
    'grid': '#BDC3C7',
    'revenue': '#2980B9',
    'interest': '#E74C3C',
    'correlation_positive': '#27AE60',
    'correlation_negative': '#C0392B',
    'neutral': '#7F8C8D'
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

# Precompute correlations for quick access
correlation_df = pd.DataFrame()
try:
    for lag in range(0, 13):
        lag_df = df.copy()
        if lag > 0:
            lag_df[f'Δ Revenue_T{lag}'] = lag_df['Δ Revenue'].shift(-lag)
            lag_df = lag_df.dropna(subset=[f'Δ Revenue_T{lag}'])
        else:
            lag_df['Δ Revenue_T0'] = lag_df['Δ Revenue']
        
        for rate_var in ['OCR', '30 day bill', 'Δ 30 day']:
            if rate_var in lag_df.columns:
                corr = lag_df[rate_var].corr(lag_df[f'Δ Revenue_T{lag}'])
                correlation_df.loc[lag, rate_var] = corr
except Exception as e:
    print(f"Error calculating correlations: {e}")

# Create the Dash app with advanced features if available
if HAS_ADVANCED_COMPONENTS:
    app = DashProxy(
        __name__, 
        external_stylesheets=[dbc.themes.FLATLY],
        transforms=[ServersideOutputTransform(), MultiplexerTransform()]
    )
else:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.title = "Advanced Interest Rate & Revenue Dashboard"

# Define time periods for dropdown
unique_years = sorted(df['Date'].dt.year.unique())
time_periods = ["All Data"]
time_periods.extend([str(year) for year in unique_years])
time_periods.extend(["Pre-COVID", "COVID and After"])

# Define time lags for dropdown
time_lags = list(range(0, 13))  # 0 to 12 months

# Create the header with title and description
header = dbc.Card(
    dbc.CardBody([
        html.H1("Interest Rate & Revenue Analysis Dashboard", className="display-4 text-center mb-3"),
        html.P(
            "Analyze the relationship between interest rates and revenue with advanced time lag analysis, "
            "forecasting, and correlation metrics.",
            className="lead text-center mb-4"
        ),
        html.Hr()
    ]),
    className="mb-4"
)

# Create the control panel
if HAS_ADVANCED_COMPONENTS:
    control_panel = dbc.Card([
        dbc.CardHeader([
            html.H3("Control Panel", className="mb-0"),
            dmc.Tooltip(
                label="Configure analysis parameters",
                children=[
                    html.I(className="fas fa-info-circle ms-2")
                ]
            )
        ], className="d-flex justify-content-between align-items-center"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Select Time Period:", className="fw-bold"),
                    dmc.Select(
                        id='time-period-dropdown',
                        data=[{"label": period, "value": period} for period in time_periods],
                        value='All Data',
                        clearable=False,
                        searchable=True,
                        style={"width": "100%"},
                        size="md"
                    ),
                ], md=6),
                dbc.Col([
                    html.P("Time Lag Analysis (Months):", className="fw-bold"),
                    dmc.Slider(
                        id='time-lag-slider',
                        min=0,
                        max=12,
                        step=1,
                        value=0,
                        marks={i: f"T+{i}" for i in range(0, 13, 3)},
                        size="md",
                        style={"width": "100%"},
                        showLabelOnHover=True,
                        labelAlwaysVisible=True
                    ),
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.P("Analysis Variables:", className="fw-bold"),
                    dmc.ChipGroup(
                        id='rate-variable-chip',
                        value='Δ 30 day',
                        children=[
                            dmc.Chip(
                                value='OCR',
                                children="OCR",
                                color="blue",
                                variant="filled",
                                size="md",
                            ),
                            dmc.Chip(
                                value='30 day bill',
                                children="30 day bill",
                                color="blue",
                                variant="filled",
                                size="md",
                            ),
                            dmc.Chip(
                                value='Δ 30 day',
                                children="Δ 30 day",
                                color="blue",
                                variant="filled",
                                size="md",
                            ),
                        ],
                    ),
                ], md=6),
                dbc.Col([
                    html.P("Chart Options:", className="fw-bold"),
                    dmc.ChipGroup(
                        id='chart-options',
                        value=['show_reg', 'show_conf'],
                        multiple=True,
                        children=[
                            dmc.Chip(
                                value='show_ma',
                                children="Show Moving Avg",
                                color="teal",
                                variant="filled",
                                size="sm",
                            ),
                            dmc.Chip(
                                value='show_reg',
                                children="Show Regression",
                                color="teal",
                                variant="filled",
                                size="sm",
                            ),
                            dmc.Chip(
                                value='show_conf',
                                children="Show Confidence Intervals",
                                color="teal",
                                variant="filled",
                                size="sm",
                            ),
                            dmc.Chip(
                                value='show_forecast',
                                children="Show Forecast",
                                color="teal",
                                variant="filled",
                                size="sm",
                            ),
                        ],
                    ),
                ], md=6),
            ], className="mb-3"),
        ]),
    ], className="mb-4")
else:
    control_panel = dbc.Card([
        dbc.CardHeader("Control Panel"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Select Time Period:", className="fw-bold"),
                    dcc.Dropdown(
                        id='time-period-dropdown',
                        options=[{'label': period, 'value': period} for period in time_periods],
                        value='All Data',
                        clearable=False
                    ),
                ], md=6),
                dbc.Col([
                    html.P("Time Lag Analysis (Months):", className="fw-bold"),
                    dcc.Slider(
                        id='time-lag-slider',
                        min=0,
                        max=12,
                        step=1,
                        value=0,
                        marks={i: f"T+{i}" for i in range(0, 13, 3)},
                    ),
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.P("Analysis Variables:", className="fw-bold"),
                    dcc.Dropdown(
                        id='rate-variable-dropdown',
                        options=[
                            {'label': 'OCR', 'value': 'OCR'},
                            {'label': '30 day bill', 'value': '30 day bill'},
                            {'label': 'Δ 30 day', 'value': 'Δ 30 day'}
                        ],
                        value='Δ 30 day',
                        clearable=False
                    ),
                ], md=6),
                dbc.Col([
                    html.P("Chart Options:", className="fw-bold"),
                    dbc.Checklist(
                        options=[
                            {"label": "Display Moving Average", "value": "show_ma"},
                            {"label": "Display Regression Line", "value": "show_reg"},
                            {"label": "Show Confidence Intervals", "value": "show_conf"},
                            {"label": "Show Forecast", "value": "show_forecast"},
                        ],
                        value=["show_reg", "show_conf"],
                        id="chart-options",
                        inline=True,
                    ),
                ], md=6),
            ], className="mb-3"),
        ]),
    ], className="mb-4")

# Create the main dashboard layout
dashboard_layout = html.Div([
    dbc.Container([
        header,
        dbc.Row([
            dbc.Col(control_panel, lg=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H3("Key Metrics", className="mb-0"),
                        html.Div(id="last-updated", className="text-muted")
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody(id="metrics-container")
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardHeader("Correlation Heatmap"),
                    dbc.CardBody(dcc.Graph(id='correlation-heatmap'))
                ])
            ], lg=8)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Interest Rate & Revenue Over Time"),
                    dbc.CardBody(dcc.Graph(id='time-series-chart', style={'height': '500px'}))
                ])
            ], lg=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Regression Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='correlation-chart', style={'height': '400px'})
                    ]),
                    dbc.CardFooter(html.Div(id='regression-output'))
                ])
            ], lg=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Advanced Analysis"),
                    dbc.CardBody(
                        dbc.Tabs([
                            dbc.Tab(dcc.Graph(id='distribution-chart'), label="Distributions"),
                            dbc.Tab(dcc.Graph(id='seasonal-decomposition'), label="Seasonal Decomposition"),
                            dbc.Tab(dcc.Graph(id='forecast-chart'), label="Forecast (ARIMA)")
                        ])
                    ),
                ])
            ], lg=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Table"),
                    dbc.CardBody([
                        html.Div(id='table-container'),
                    ]),
                    dbc.CardFooter([
                        dbc.Row([
                            dbc.Col([
                                html.P("Export Data:"),
                                html.Div([
                                    dbc.Button("Export to CSV", id="btn-csv", color="primary", className="me-2"),
                                    dbc.Button("Export to Excel", id="btn-excel", color="success", className="me-2"),
                                    Download(id="download-dataframe-csv"),
                                    Download(id="download-dataframe-excel"),
                                ]) if HAS_ADVANCED_COMPONENTS else html.P("Install dash-extensions for export functionality")
                            ])
                        ])
                    ]) if HAS_ADVANCED_COMPONENTS else None
                ])
            ], lg=12)
        ]),
    ], fluid=True, className="mt-4 mb-4")
])

# Set the app layout
app.layout = dashboard_layout

# Create callback for rate variable dropdown/chipgroup if advanced components are available
if HAS_ADVANCED_COMPONENTS:
    @app.callback(
        Output('rate-variable-dropdown', 'value', allow_duplicate=True),
        Input('rate-variable-chip', 'value'),
        prevent_initial_call=True
    )
    def sync_rate_var_chip(value):
        return value

    @app.callback(
        Output('rate-variable-chip', 'value', allow_duplicate=True),
        Input('rate-variable-dropdown', 'value'),
        prevent_initial_call=True
    )
    def sync_rate_var_dropdown(value):
        return value

# Callback for key metrics cards
@app.callback(
    Output('metrics-container', 'children'),
    [Input('time-period-dropdown', 'value'),
     Input('time-lag-slider', 'value'),
     Input('rate-variable-dropdown', 'value') if not HAS_ADVANCED_COMPONENTS else Input('rate-variable-chip', 'value') if HAS_ADVANCED_COMPONENTS else Input('rate-variable-dropdown', 'value')]
)
def update_metrics(time_period, time_lag, rate_variable):
    # Filter the data based on the selected time period
    filtered_df = df.copy()
    
    if time_period == "Pre-COVID":
        filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
    elif time_period == "COVID and After":
        filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
    elif time_period != "All Data":
        year = int(time_period)
        filtered_df = filtered_df[filtered_df['Date'].dt.year == year]
    
    # Calculate metrics
    try:
        avg_revenue = filtered_df['Total'].mean()
        revenue_growth = filtered_df['Δ Revenue'].mean() * 100
        avg_rate = filtered_df[rate_variable].mean() * 100
        
        # Calculate correlation based on time lag
        target_col = f'Δ Revenue_T{time_lag}'
        if target_col not in filtered_df.columns and time_lag > 0:
            filtered_df[target_col] = filtered_df['Δ Revenue'].shift(-time_lag)
        
        # Only include rows with valid data
        corr_df = filtered_df.dropna(subset=[rate_variable, target_col if time_lag > 0 else 'Δ Revenue'])
        if len(corr_df) > 3:  # Need at least 3 points for correlation
            correlation = corr_df[rate_variable].corr(corr_df[target_col if time_lag > 0 else 'Δ Revenue'])
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
        
        # Create the metrics cards
        if HAS_ADVANCED_COMPONENTS:
            metrics_cards = dbc.Row([
                dbc.Col(dmc.Card([
                    dmc.CardSection(
                        dmc.Text("Average Revenue", weight=500, color="dimmed", size="sm"),
                        withBorder=True,
                        inheritPadding=True,
                        py="xs",
                    ),
                    dmc.Group([
                        dmc.Text(f"${avg_revenue:,.0f}", weight=700, size="xl"),
                    ], position="apart", mt="md", mb="xs"),
                ]), width=3),
                dbc.Col(dmc.Card([
                    dmc.CardSection(
                        dmc.Text("Revenue Growth", weight=500, color="dimmed", size="sm"),
                        withBorder=True,
                        inheritPadding=True,
                        py="xs",
                    ),
                    dmc.Group([
                        dmc.Text(
                            f"{revenue_growth:.2f}%", 
                            weight=700, 
                            size="xl",
                            color="green" if revenue_growth > 0 else "red"
                        ),
                    ], position="apart", mt="md", mb="xs"),
                ]), width=3),
                dbc.Col(dmc.Card([
                    dmc.CardSection(
                        dmc.Text(f"Average {rate_variable}", weight=500, color="dimmed", size="sm"),
                        withBorder=True,
                        inheritPadding=True,
                        py="xs",
                    ),
                    dmc.Group([
                        dmc.Text(f"{avg_rate:.2f}%", weight=700, size="xl"),
                    ], position="apart", mt="md", mb="xs"),
                ]), width=3),
                dbc.Col(dmc.Card([
                    dmc.CardSection(
                        dmc.Text(f"Correlation (T+{time_lag})", weight=500, color="dimmed", size="sm"),
                        withBorder=True,
                        inheritPadding=True,
                        py="xs",
                    ),
                    dmc.Group([
                        dmc.Text(
                            f"{correlation:.3f}" if not np.isnan(correlation) else "Insufficient data", 
                            weight=700, 
                            size="xl",
                            color=corr_color
                        ),
                    ], position="apart", mt="md", mb="xs"),
                ]), width=3),
            ])
        else:
            metrics_cards = dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H4("Average Revenue"),
                        html.H2(f"${avg_revenue:,.0f}", className="text-primary"),
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H4("Revenue Growth"),
                        html.H2(
                            f"{revenue_growth:.2f}%", 
                            className="text-success" if revenue_growth > 0 else "text-danger"
                        ),
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H4(f"Average {rate_variable}"),
                        html.H2(f"{avg_rate:.2f}%", className="text-info"),
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H4(f"Correlation (T+{time_lag})"),
                        html.H2(
                            f"{correlation:.3f}" if not np.isnan(correlation) else "Insufficient data", 
                            className=f"text-{corr_color}"
                        ),
                    ])
                ]), width=3),
            ])
        
        return metrics_cards
    
    except Exception as e:
        print(f"Error updating metrics: {e}")
        return html.Div(f"Error calculating metrics: {str(e)}")

# Callback for the correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('time-period-dropdown', 'value'),
     Input('rate-variable-dropdown', 'value') if not HAS_ADVANCED_COMPONENTS else Input('rate-variable-chip', 'value') if HAS_ADVANCED_COMPONENTS else Input('rate-variable-dropdown', 'value')]
)
def update_correlation_heatmap(time_period, rate_variable):
    try:
        # Filter the data based on the selected time period
        filtered_df = df.copy()
        
        if time_period == "Pre-COVID":
            filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
        elif time_period == "COVID and After":
            filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
        elif time_period != "All Data":
            year = int(time_period)
            filtered_df = filtered_df[filtered_df['Date'].dt.year == year]
        
        # Create lag variables
        filtered_df['Δ Revenue_T0'] = filtered_df['Δ Revenue']  # No lag
        for lag in range(1, 13):
            filtered_df[f'Δ Revenue_T{lag}'] = filtered_df['Δ Revenue'].shift(-lag)
        
        # Compute correlation matrix
        corr_cols = [rate_variable] + [f'Δ Revenue_T{lag}' for lag in range(0, 13)]
        
        # Make sure all columns exist
        for col in corr_cols:
            if col not in filtered_df.columns:
                print(f"Column {col} not found in dataframe, creating it with NaN values")
                filtered_df[col] = np.nan
        
        corr_df = filtered_df[corr_cols].dropna().corr()
        
        # Extract first row/column (correlations with selected rate variable)
        corr_values = []
        for lag in range(0, 13):
            lag_col = f'Δ Revenue_T{lag}'
            if lag_col in corr_df.columns and rate_variable in corr_df.index:
                corr_values.append(corr_df.loc[rate_variable, lag_col])
            else:
                corr_values.append(np.nan)
        
        # Create the heatmap figure
        fig = go.Figure()
        
        # Add the heatmap trace
        fig.add_trace(go.Heatmap(
            z=[corr_values],
            x=[f'T+{lag}' for lag in range(0, 13)],
            y=[rate_variable],
            colorscale=[
                [0.0, 'rgb(165,0,38)'],
                [0.1, 'rgb(215,48,39)'],
                [0.2, 'rgb(244,109,67)'],
                [0.3, 'rgb(253,174,97)'],
                [0.4, 'rgb(254,224,144)'],
                [0.5, 'rgb(255,255,191)'],
                [0.6, 'rgb(224,243,248)'],
                [0.7, 'rgb(171,217,233)'],
                [0.8, 'rgb(116,173,209)'],
                [0.9, 'rgb(69,117,180)'],
                [1.0, 'rgb(49,54,149)']
            ],
            colorbar=dict(
                title="Correlation",
                titleside="right"
            ),
            zmin=-1, zmax=1
        ))
        
        # Add correlation values as text
        for i, lag in enumerate(range(0, 13)):
            fig.add_annotation(
                x=f'T+{lag}',
                y=rate_variable,
                text=f"{corr_values[i]:.3f}",
                showarrow=False,
                font=dict(
                    color="white" if abs(corr_values[i]) > 0.4 else "black"
                )
            )
        
        # Find the maximum correlation lag
        max_corr_idx = np.nanargmax(np.abs(corr_values))
        max_corr_lag = max_corr_idx
        
        # Add a box around the maximum correlation
        fig.add_shape(
            type="rect",
            x0=max_corr_lag-0.5,
            x1=max_corr_lag+0.5,
            y0=-0.5,
            y1=0.5,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
        )
        
        fig.update_layout(
            title=f"Correlation between {rate_variable} and Revenue Change by Time Lag",
            xaxis_title="Time Lag (months)",
            margin=dict(l=20, r=20, t=60, b=20),
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
            showarrow=False
        )
        return fig

# Create the main callbacks for visualizations
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('correlation-chart', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('seasonal-decomposition', 'figure'),
     Output('forecast-chart', 'figure'),
     Output('regression-output', 'children'),
     Output('table-container', 'children'),
     Output('last-updated', 'children')],
    [Input('time-period-dropdown', 'value'),
     Input('time-lag-slider', 'value'),
     Input('rate-variable-dropdown', 'value') if not HAS_ADVANCED_COMPONENTS else Input('rate-variable-chip', 'value') if HAS_ADVANCED_COMPONENTS else Input('rate-variable-dropdown', 'value'),
     Input('chart-options', 'value')]
)
def update_charts(time_period, time_lag, rate_variable, chart_options):
    # Get current time for last updated
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_updated = f"Last updated: {current_time}"
    
    # Filter the data based on the selected time period
    filtered_df = df.copy()
    
    if time_period == "Pre-COVID":
        filtered_df = filtered_df[filtered_df['Date'] < COVID_START]
    elif time_period == "COVID and After":
        filtered_df = filtered_df[filtered_df['Date'] >= COVID_START]
    elif time_period != "All Data":
        year = int(time_period)
        filtered_df = filtered_df[filtered_df['Date'].dt.year == year]
    
    # Create a lag-adjusted dataframe for analysis
    lag_df = filtered_df.copy()
    
    try:
        if time_lag > 0:
            # Make sure Δ Revenue exists
            if 'Δ Revenue' not in lag_df.columns:
                print("Calculating Δ Revenue for lag analysis")
                lag_df['Δ Revenue'] = lag_df['Total'].pct_change()
                
            # Shift the revenue changes back by the selected lag
            lag_df['Lag Revenue Change'] = lag_df['Δ Revenue'].shift(-time_lag)
            # Only drop rows if we need to (keep more data for visualization)
            lag_df_for_regression = lag_df.dropna(subset=['Lag Revenue Change'])
        else:
            lag_df['Lag Revenue Change'] = lag_df['Δ Revenue']
            lag_df_for_regression = lag_df
    except Exception as e:
        print(f"Error in time lag calculation: {e}")
        lag_df['Lag Revenue Change'] = lag_df['Δ Revenue']
        lag_df_for_regression = lag_df
    
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
        if COVID_START in filtered_df['Date'].values:
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
        target_column = 'Lag Revenue Change' if time_lag > 0 else 'Δ Revenue'
        valid_scatter_data = lag_df.dropna(subset=[rate_variable, target_column])
        
        # Only try to add trendline if we have sufficient data
        use_trendline = "show_reg" in chart_options and len(valid_scatter_data) >= 3
        
        fig_correlation = px.scatter(
            valid_scatter_data, 
            x=rate_variable, 
            y=target_column,
            trendline='ols' if use_trendline else None,
            labels={
                rate_variable: rate_variable,
                target_column: f"Revenue Change (T+{time_lag})" if time_lag > 0 else "Revenue Change"
            },
            title=f"Correlation between {rate_variable} and Revenue Change (T+{time_lag})" if time_lag > 0 else f"Correlation between {rate_variable} and Revenue Change",
            template="plotly_white",
            color_discrete_sequence=[COLOR_SCHEME['accent']]
        )
        
        # Add confidence intervals if selected
        if "show_conf" in chart_options and use_trendline and len(valid_scatter_data) >= 5:
            # Perform regression to get confidence intervals
            X = sm.add_constant(valid_scatter_data[rate_variable])
            y = valid_scatter_data[target_column]
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
        valid_rate_data = lag_df.dropna(subset=[rate_variable])
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
            
            # Add KDE curve
            if len(valid_rate_data) >= 10:  # Need enough data for KDE
                from scipy.stats import gaussian_kde
                
                kde = gaussian_kde(valid_rate_data[rate_variable].dropna())
                x_range = np.linspace(valid_rate_data[rate_variable].min(), valid_rate_data[rate_variable].max(), 100)
                y_range = kde(x_range) * len(valid_rate_data) * (valid_rate_data[rate_variable].max() - valid_rate_data[rate_variable].min()) / 10
                
                fig_distribution.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        name='KDE',
                        line=dict(color='black', width=2),
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
        
        # Add histogram for revenue change
        target_column = 'Lag Revenue Change' if time_lag > 0 else 'Δ Revenue'
        if target_column in lag_df.columns:
            valid_revenue_data = lag_df.dropna(subset=[target_column])
            if len(valid_revenue_data) > 0:
                # Add histogram
                fig_distribution.add_trace(
                    go.Histogram(
                        x=valid_revenue_data[target_column],
                        name=f"Revenue Change (T+{time_lag})" if time_lag > 0 else "Revenue Change",
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
                
                # Add KDE curve
                if len(valid_revenue_data) >= 10:  # Need enough data for KDE
                    kde = gaussian_kde(valid_revenue_data[target_column].dropna())
                    x_range = np.linspace(valid_revenue_data[target_column].min(), valid_revenue_data[target_column].max(), 100)
                    y_range = kde(x_range) * len(valid_revenue_data) * (valid_revenue_data[target_column].max() - valid_revenue_data[target_column].min()) / 10
                    
                    fig_distribution.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            name='KDE',
                            line=dict(color='black', width=2),
                            hoverinfo='skip'
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
        
        fig_distribution.update_xaxes(
            title_text=f"Revenue Change (T+{time_lag})" if time_lag > 0 else "Revenue Change",
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
    
    # Create ARIMA forecast chart
    try:
        if "show_forecast" in chart_options and len(filtered_df) >= 24 and HAS_ADVANCED_ANALYSIS:
            # Use revenue for forecasting
            ts_data = filtered_df.set_index('Date')['Total']
            
            # Fit auto ARIMA model
            model = pm.auto_arima(
                ts_data,
                seasonal=True,
                m=12,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True
            )
            
            # Generate forecasts
            n_periods = 12  # 12 months forecast
            forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
            
            # Create forecast index
            last_date = ts_data.index[-1]
            forecast_idx = pd.date_range(start=last_date, periods=n_periods+1, freq='MS')[1:]
            
            # Create the chart
            fig_forecast = go.Figure()
            
            # Add historical data
            fig_forecast.add_trace(
                go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color=COLOR_SCHEME['primary'], width=2)
                )
            )
            
            # Add forecast
            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_idx,
                    y=forecast,
                    mode='lines',
                    name='Forecast',
                    line=dict(color=COLOR_SCHEME['secondary'], width=2, dash='dash')
                )
            )
            
            # Add confidence intervals
            fig_forecast.add_trace(
                go.Scatter(
                    x=np.concatenate([forecast_idx, forecast_idx[::-1]]),
                    y=np.concatenate([conf_int[:, 0], conf_int[::-1, 1]]),
                    fill='toself',
                    fillcolor='rgba(0, 100, 80, 0.2)',
                    line=dict(color='rgba(0, 0, 0, 0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    name='95% Confidence Interval'
                )
            )
            
            # Update layout
            fig_forecast.update_layout(
                title=dict(
                    text=f"Revenue Forecast (ARIMA {model.order}, Seasonal {model.seasonal_order})",
                    font=dict(size=16, color=COLOR_SCHEME['primary'])
                ),
                xaxis_title="Date",
                yaxis_title="Revenue",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor=COLOR_SCHEME['background'],
                paper_bgcolor=COLOR_SCHEME['background'],
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # Add annotation for forecast start
            fig_forecast.add_shape(
                type="line",
                x0=last_date,
                x1=last_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="black", width=1, dash="dot"),
            )
            
            fig_forecast.add_annotation(
                x=last_date,
                y=1,
                yref="paper",
                text="Forecast Start",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40
            )
        else:
            # Not enough data or forecast not requested
            if "show_forecast" not in chart_options:
                message = "Enable 'Show Forecast' option to view ARIMA forecasts"
            elif not HAS_ADVANCED_ANALYSIS:
                message = "Install pmdarima package to enable ARIMA forecasting"
            else:
                message = "Insufficient data for forecasting (need at least 24 months)"
                
            fig_forecast = go.Figure()
            fig_forecast.add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
    except Exception as e:
        print(f"Error creating forecast chart: {e}")
        # Create empty figure with error message
        fig_forecast = go.Figure()
        fig_forecast.add_annotation(
            text=f"Error creating forecast chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    
    # Perform regression analysis
    try:
        # Check if we have enough data
        if len(lag_df_for_regression) < 3:
            regression_output = html.Div([
                html.H5("Insufficient Data", className="text-warning"),
                html.P("Not enough data points for regression analysis.")
            ])
        else:
            # Check for missing values
            target_column = 'Lag Revenue Change' if time_lag > 0 else 'Δ Revenue'
            valid_data = lag_df_for_regression.dropna(subset=[rate_variable, target_column])
            
            if len(valid_data) < 3:
                regression_output = html.Div([
                    html.H5("Insufficient Valid Data", className="text-warning"),
                    html.P("Not enough valid data points after removing missing values.")
                ])
            else:
                # Perform regression
                X = valid_data[rate_variable].values.reshape(-1, 1)
                y = valid_data[target_column].values
                
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()
                
                # Calculate advanced metrics for predictive power if available
                predictive_power = None
                if HAS_ADVANCED_ANALYSIS:
                    try:
                        pp_score = pps.score(
                            pd.DataFrame({rate_variable: valid_data[rate_variable]}),
                            pd.DataFrame({target_column: valid_data[target_column]})
                        )
                        predictive_power = pp_score['ppscore'][0]
                    except Exception as pp_error:
                        print(f"Error calculating predictive power score: {pp_error}")
                
                # Create the regression output with enhanced styling
                if HAS_ADVANCED_COMPONENTS:
                    regression_output = dbc.Row([
                        dbc.Col([
                            dmc.Paper(
                                dmc.Stack([
                                    dmc.Text("Regression Results", weight=700, size="lg"),
                                    dmc.Divider(),
                                    dmc.Grid([
                                        dmc.Col([
                                            dmc.Text("R-squared:", weight=700),
                                            dmc.Text("Adjusted R-squared:", weight=700),
                                            dmc.Text("p-value:", weight=700),
                                            dmc.Text("Coefficient:", weight=700),
                                            dmc.Text("Predictive Power Score:", weight=700) if predictive_power is not None else None,
                                        ], span=6),
                                        dmc.Col([
                                            dmc.Text(f"{model.rsquared:.4f}", color="blue" if model.rsquared > 0.3 else "gray"),
                                            dmc.Text(f"{model.rsquared_adj:.4f}", color="blue" if model.rsquared_adj > 0.3 else "gray"),
                                            dmc.Text(f"{model.f_pvalue:.4f}", color="green" if model.f_pvalue < 0.05 else "red"),
                                            dmc.Text(f"{model.params[1]:.4f}", color="blue"),
                                            dmc.Text(f"{predictive_power:.4f}", color="blue" if predictive_power and predictive_power > 0.2 else "gray") if predictive_power is not None else None,
                                        ], span=6),
                                    ]),
                                    dmc.Divider(),
                                    dmc.Text("Interpretation:", weight=700),
                                    dmc.Text(
                                        f"The model explains {model.rsquared:.1%} of the variance in revenue change. "
                                        f"A 1% {'increase' if model.params[1] > 0 else 'decrease'} in {rate_variable} is "
                                        f"associated with a {abs(model.params[1]):.2%} {'increase' if model.params[1] > 0 else 'decrease'} "
                                        f"in revenue change after {time_lag} month{'s' if time_lag != 1 else ''}."
                                    ),
                                    dmc.Text(
                                        f"The relationship is statistically {'significant' if model.f_pvalue < 0.05 else 'not significant'} "
                                        f"(p = {model.f_pvalue:.4f})."
                                    ),
                                ], spacing="sm"),
                                p="md",
                                radius="md",
                                withBorder=True,
                                shadow="sm",
                            )
                        ], md=12)
                    ])
                else:
                    regression_output = html.Div([
                        html.H5(f"Regression Results (T+{time_lag})" if time_lag > 0 else "Regression Results"),
                        dbc.Row([
                            dbc.Col([
                                html.P(f"R-squared: {model.rsquared:.4f}", className="text-primary" if model.rsquared > 0.3 else ""),
                                html.P(f"Adjusted R-squared: {model.rsquared_adj:.4f}", className="text-primary" if model.rsquared_adj > 0.3 else ""),
                            ], md=6),
                            dbc.Col([
                                html.P(f"p-value: {model.f_pvalue:.4f}", className="text-success" if model.f_pvalue < 0.05 else "text-danger"),
                                html.P(f"Coefficient: {model.params[1]:.4f}"),
                            ], md=6),
                        ]),
                        html.Hr(),
                        html.P(
                            f"The model explains {model.rsquared:.1%} of the variance in revenue change. "
                            f"A 1% {'increase' if model.params[1] > 0 else 'decrease'} in {rate_variable} is "
                            f"associated with a {abs(model.params[1]):.2%} {'increase' if model.params[1] > 0 else 'decrease'} "
                            f"in revenue change after {time_lag} month{'s' if time_lag != 1 else ''}. "
                            f"The relationship is statistically {'significant' if model.f_pvalue < 0.05 else 'not significant'} "
                            f"(p = {model.f_pvalue:.4f})."
                        ),
                    ])
    except Exception as e:
        print(f"Error in regression analysis: {e}")
        regression_output = html.Div([
            html.H5("Regression Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ])
    
    # Prepare data for the table with enhanced styling
    try:
        table_df = lag_df.copy()  # Don't drop NAs to show all data
        
        # Determine which columns to include
        if time_lag > 0:
            table_columns = ['Date', 'Total', rate_variable, 'Δ Revenue']
            if 'Lag Revenue Change' in table_df.columns:
                table_columns.append('Lag Revenue Change')
        else:
            table_columns = ['Date', 'Total', rate_variable, 'Δ Revenue']
        
        # Only include columns that exist in the dataframe
        table_columns = [col for col in table_columns if col in table_df.columns]
        
        # Generate table data
        table_data = table_df[table_columns].to_dict('records')
        
        # Create table component based on available packages
        if HAS_ADVANCED_COMPONENTS:
            # Use AG Grid for advanced table
            table_component = dag.AgGrid(
                id="data-grid",
                rowData=table_data,
                columnDefs=[
                    {"headerName": "Date", "field": "Date", "type": "dateColumn", "filter": "agDateColumnFilter", "sortable": True},
                    {"headerName": "Revenue", "field": "Total", "type": "numericColumn", "filter": "agNumberColumnFilter", "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"}, "sortable": True},
                    {"headerName": rate_variable, "field": rate_variable, "type": "numericColumn", "filter": "agNumberColumnFilter", "valueFormatter": {"function": "d3.format('.2%')(params.value)"}, "sortable": True},
                    {"headerName": "Revenue Change", "field": "Δ Revenue", "type": "numericColumn", "filter": "agNumberColumnFilter", "valueFormatter": {"function": "d3.format('.2%')(params.value)"}, "sortable": True},
                ] + (
                    [{"headerName": f"Revenue Change (T+{time_lag})", "field": "Lag Revenue Change", "type": "numericColumn", "filter": "agNumberColumnFilter", "valueFormatter": {"function": "d3.format('.2%')(params.value)"}, "sortable": True}] 
                    if time_lag > 0 and "Lag Revenue Change" in table_columns else []
                ),
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                    "filter": True,
                },
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 10,
                    "domLayout": "autoHeight",
                },
                style={"height": "auto", "width": "100%"},
                className="ag-theme-alpine",
            )
        else:
            # Use basic Dash datatable
            formatted_columns = []
            for col in table_columns:
                if col == 'Date':
                    formatted_columns.append({"name": "Date", "id": "Date", "type": "datetime"})
                elif col in ['Δ Revenue', 'Δ 30 day', 'OCR', '30 day bill', 'Lag Revenue Change']:
                    formatted_columns.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ".2%"}})
                else:
                    formatted_columns.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ",.2f"}})
            
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
    
    return fig_time_series, fig_correlation, fig_distribution, fig_seasonal, fig_forecast, regression_output, table_component, last_updated

# Add callbacks for data export if advanced components are available
if HAS_ADVANCED_COMPONENTS:
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("btn-csv", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_csv(n_clicks):
        return dcc.send_data_frame(df.to_csv, "interest_rate_revenue_data.csv")

    @app.callback(
        Output("download-dataframe-excel", "data"),
        Input("btn-excel", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_excel(n_clicks):
        return dcc.send_data_frame(df.to_excel, "interest_rate_revenue_data.xlsx", sheet_name="Data")

if __name__ == '__main__':
    app.run_server(debug=True)