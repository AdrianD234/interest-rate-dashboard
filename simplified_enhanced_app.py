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

# Create Δ Revenue time lags for all months ahead of time
df['Δ Revenue_T0'] = df['Δ Revenue']  # No lag
for lag in range(1, 13):
    df[f'Δ Revenue_T{lag}'] = df['Δ Revenue'].shift(-lag)

# Create the Dash app
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
                            dbc.Tab(dcc.Graph(id='forecast-chart'), label="Forecast")
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
                ])
            ], lg=12)
        ]),
    ], fluid=True, className="mt-4 mb-4")
])

# Set the app layout
app.layout = dashboard_layout

# Callback for key metrics cards
@app.callback(
    Output('metrics-container', 'children'),
    [Input('time-period-dropdown', 'value'),
     Input('time-lag-slider', 'value'),
     Input('rate-variable-dropdown', 'value')]
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
        
        # Only include rows with valid data
        corr_df = filtered_df.dropna(subset=[rate_variable, target_col])
        if len(corr_df) > 3:  # Need at least 3 points for correlation
            correlation = corr_df[rate_variable].corr(corr_df[target_col])
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
     Input('rate-variable-dropdown', 'value')]
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
        
        # Make sure all necessary lag columns exist
        for lag in range(0, 13):
            if f'Δ Revenue_T{lag}' not in filtered_df.columns:
                filtered_df[f'Δ Revenue_T{lag}'] = np.nan
        
        # Compute correlation values for all lags
        corr_values = []
        for lag in range(0, 13):
            lag_col = f'Δ Revenue_T{lag}'
            # Calculate correlation between rate variable and lagged revenue change
            valid_data = filtered_df.dropna(subset=[rate_variable, lag_col])
            if len(valid_data) > 2:  # Need at least 3 points for correlation
                corr = valid_data[rate_variable].corr(valid_data[lag_col])
                corr_values.append(corr)
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
            if not np.isnan(corr_values[i]):
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
        if any(not np.isnan(x) for x in corr_values):
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
     Input('rate-variable-dropdown', 'value'),
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
        # Ensure the target column exists
        target_column = f'Δ Revenue_T{time_lag}'
        if target_column not in lag_df.columns:
            print(f"Creating lag column: {target_column}")
            if time_lag == 0:
                lag_df[target_column] = lag_df['Δ Revenue']
            else:
                lag_df[target_column] = lag_df['Δ Revenue'].shift(-time_lag)
        
        # Alias for simpler referencing
        lag_df['Lag Revenue Change'] = lag_df[target_column]
        
        # Only drop rows if we need to (keep more data for visualization)
        lag_df_for_regression = lag_df.dropna(subset=['Lag Revenue Change'])
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
        target_column = 'Lag Revenue Change'
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
        
        # Add histogram for revenue change
        target_column = 'Lag Revenue Change'
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
    
    # Create forecast chart - simple version
    try:
        if "show_forecast" in chart_options and len(filtered_df) >= 12:
            # Use simple linear trend forecasting
            # Get revenue data
            y = filtered_df['Total'].values
            X = np.arange(len(y)).reshape(-1, 1)
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 12 months
            X_future = np.arange(len(y), len(y) + 12).reshape(-1, 1)
            forecast = model.predict(X_future)
            
            # Calculate confidence intervals (simple approach)
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            std_error = np.sqrt(mse)
            
            # 95% CI is roughly ±2 standard errors
            forecast_lower = forecast - 2 * std_error
            forecast_upper = forecast + 2 * std_error
            
            # Create date index for forecast
            last_date = filtered_df['Date'].iloc[-1]
            date_range = pd.date_range(start=last_date, periods=13, freq='MS')[1:]
            
            # Create the chart
            fig_forecast = go.Figure()
            
            # Add historical data
            fig_forecast.add_trace(
                go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df['Total'],
                    mode='lines',
                    name='Historical',
                    line=dict(color=COLOR_SCHEME['primary'], width=2)
                )
            )
            
            # Add forecast
            fig_forecast.add_trace(
                go.Scatter(
                    x=date_range,
                    y=forecast,
                    mode='lines',
                    name='Forecast',
                    line=dict(color=COLOR_SCHEME['secondary'], width=2, dash='dash')
                )
            )
            
            # Add confidence intervals
            fig_forecast.add_trace(
                go.Scatter(
                    x=np.concatenate([date_range, date_range[::-1]]),
                    y=np.concatenate([forecast_lower, forecast_upper[::-1]]),
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
                    text="Revenue Forecast (Linear Trend)",
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
                message = "Enable 'Show Forecast' option to view forecasts"
            else:
                message = "Insufficient data for forecasting (need at least 12 months)"
                
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
            target_column = 'Lag Revenue Change'
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
                
                # Create the regression output with enhanced styling
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
        
        # Create formatted columns for the table
        formatted_columns = []
        for col in table_columns:
            if col == 'Date':
                formatted_columns.append({"name": "Date", "id": "Date", "type": "datetime"})
            elif col in ['Δ Revenue', 'Δ 30 day', 'OCR', '30 day bill', 'Lag Revenue Change']:
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
    
    return fig_time_series, fig_correlation, fig_distribution, fig_seasonal, fig_forecast, regression_output, table_component, last_updated

if __name__ == '__main__':
    app.run_server(debug=True)