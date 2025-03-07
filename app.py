import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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
        df['Δ Revenue'] = df['Total'].pct_change() * 100
        
    if '30 day bill' in df.columns and 'Δ 30 day' not in df.columns:
        print("Calculating Δ 30 day")
        df['Δ 30 day'] = df['30 day bill'].pct_change() * 100
    
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
        if df.columns[0] != 'checkDate':
            df = df.rename(columns={df.columns[0]: 'checkDate'})
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

# Set time periods for COVID analysis
COVID_START = pd.to_datetime('2020-03-01')  # March 2020 as approximate COVID start

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Interest Rate & Revenue Dashboard"

# Define time periods for dropdown
unique_years = sorted(df['Date'].dt.year.unique())
time_periods = ["All Data"]
time_periods.extend([str(year) for year in unique_years])
time_periods.extend(["Pre-COVID", "COVID and After"])

# Define time lags for dropdown
time_lags = list(range(0, 13))  # 0 to 12 months

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Interest Rate & Revenue Analysis Dashboard", 
                   style={'textAlign': 'center', 'marginTop': 20, 'marginBottom': 20})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Control Panel"),
                dbc.CardBody([
                    html.P("Select Time Period:"),
                    dcc.Dropdown(
                        id='time-period-dropdown',
                        options=[{'label': period, 'value': period} for period in time_periods],
                        value='All Data',
                        clearable=False
                    ),
                    html.Br(),
                    
                    html.P("Time Lag Analysis (Months):"),
                    dcc.Dropdown(
                        id='time-lag-dropdown',
                        options=[{'label': f"T+{lag}" if lag > 0 else "No Lag", 'value': lag} for lag in time_lags],
                        value=0,
                        clearable=False
                    ),
                    html.Br(),
                    
                    html.P("Analysis Variables:"),
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
                    html.Br(),
                    
                    html.P("Advanced Options:"),
                    dbc.Checklist(
                        options=[
                            {"label": "Display 12-month Moving Average", "value": "show_ma"},
                            {"label": "Display Regression Line", "value": "show_reg"},
                        ],
                        value=["show_reg"],
                        id="advanced-options",
                        inline=True,
                    ),
                ])
            ]),
            html.Br(),
            
            dbc.Card([
                dbc.CardHeader("Regression Analysis"),
                dbc.CardBody([
                    html.Div(id='regression-output')
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Interest Rate & Revenue Over Time"),
                dbc.CardBody([
                    dcc.Graph(id='time-series-chart', style={'height': '400px'})
                ])
            ]),
            html.Br(),
            
            dbc.Card([
                dbc.CardHeader("Correlation Analysis"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(
                            dcc.Graph(id='correlation-chart', style={'height': '400px'}),
                            label="Scatter Plot"
                        ),
                        dbc.Tab(
                            dcc.Graph(id='distribution-chart', style={'height': '400px'}),
                            label="Distributions"
                        ),
                    ])
                ])
            ])
        ], width=9)
    ]),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='data-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        page_size=10
                    )
                ])
            ])
        ])
    ])
], fluid=True)


@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('correlation-chart', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('regression-output', 'children'),
     Output('data-table', 'data'),
     Output('data-table', 'columns')],
    [Input('time-period-dropdown', 'value'),
     Input('time-lag-dropdown', 'value'),
     Input('rate-variable-dropdown', 'value'),
     Input('advanced-options', 'value')]
)
def update_charts(time_period, time_lag, rate_variable, advanced_options):
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
    
    # Create the time series chart
    fig_time_series = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add revenue to the primary y-axis
    fig_time_series.add_trace(
        go.Scatter(
            x=filtered_df['Date'], 
            y=filtered_df['Total'], 
            name='Revenue',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add selected interest rate to the secondary y-axis
    fig_time_series.add_trace(
        go.Scatter(
            x=filtered_df['Date'], 
            y=filtered_df[rate_variable], 
            name=rate_variable,
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    # Add moving average if selected
    if "show_ma" in advanced_options:
        filtered_df['Revenue_MA'] = filtered_df['Total'].rolling(window=12).mean()
        fig_time_series.add_trace(
            go.Scatter(
                x=filtered_df['Date'], 
                y=filtered_df['Revenue_MA'], 
                name='Revenue 12M MA',
                line=dict(color='darkblue', dash='dash')
            ),
            secondary_y=False
        )
    
    fig_time_series.update_layout(
        title_text="Revenue and Interest Rate Over Time",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_time_series.update_xaxes(title_text="Date")
    fig_time_series.update_yaxes(title_text="Revenue", secondary_y=False)
    fig_time_series.update_yaxes(title_text=rate_variable, secondary_y=True)
    
    # Create the correlation scatter plot
    try:
        target_column = 'Lag Revenue Change' if time_lag > 0 else 'Δ Revenue'
        valid_scatter_data = lag_df.dropna(subset=[rate_variable, target_column])
        
        # Only try to add trendline if we have sufficient data
        use_trendline = "show_reg" in advanced_options and len(valid_scatter_data) >= 3
        
        fig_correlation = px.scatter(
            valid_scatter_data, 
            x=rate_variable, 
            y=target_column,
            trendline='ols' if use_trendline else None,
            labels={
                rate_variable: rate_variable,
                target_column: f"Revenue Change (T+{time_lag})" if time_lag > 0 else "Revenue Change"
            },
            title=f"Correlation between {rate_variable} and Revenue Change (T+{time_lag})" if time_lag > 0 else f"Correlation between {rate_variable} and Revenue Change"
        )
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        # Create empty figure with error message
        fig_correlation = px.scatter(
            x=[0], y=[0],
            labels={'x': 'Error', 'y': 'Error'},
            title="Error creating scatter plot"
        )
        fig_correlation.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    
    # Create the distribution chart
    try:
        fig_distribution = make_subplots(rows=2, cols=1, shared_xaxes=False)
        
        # Add histogram for rate variable
        valid_rate_data = lag_df.dropna(subset=[rate_variable])
        if len(valid_rate_data) > 0:
            fig_distribution.add_trace(
                go.Histogram(
                    x=valid_rate_data[rate_variable],
                    name=rate_variable,
                    opacity=0.7,
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # Add histogram for revenue change
        target_column = 'Lag Revenue Change' if time_lag > 0 else 'Δ Revenue'
        if target_column in lag_df.columns:
            valid_revenue_data = lag_df.dropna(subset=[target_column])
            if len(valid_revenue_data) > 0:
                fig_distribution.add_trace(
                    go.Histogram(
                        x=valid_revenue_data[target_column],
                        name=f"Revenue Change (T+{time_lag})" if time_lag > 0 else "Revenue Change",
                        opacity=0.7,
                        marker_color='blue'
                    ),
                    row=2, col=1
                )
        
        fig_distribution.update_layout(
            title_text="Distribution of Variables",
            showlegend=True,
            height=400
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
        fig_distribution.update_layout(
            title_text="Distribution Error",
            height=400
        )
    
    # Perform regression analysis
    try:
        # Check if we have enough data
        if len(lag_df_for_regression) < 3:
            regression_output = html.Div([
                html.H5("Insufficient Data"),
                html.P("Not enough data points for regression analysis.")
            ])
        else:
            # Check for missing values
            target_column = 'Lag Revenue Change' if time_lag > 0 else 'Δ Revenue'
            valid_data = lag_df_for_regression.dropna(subset=[rate_variable, target_column])
            
            if len(valid_data) < 3:
                regression_output = html.Div([
                    html.H5("Insufficient Valid Data"),
                    html.P("Not enough valid data points after removing missing values.")
                ])
            else:
                # Perform regression
                X = valid_data[rate_variable].values.reshape(-1, 1)
                y = valid_data[target_column].values
                
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                
                # Create the regression output
                regression_output = html.Div([
                    html.H5(f"Regression Results (T+{time_lag})" if time_lag > 0 else "Regression Results"),
                    html.P(f"R-squared: {model.rsquared:.4f}"),
                    html.P(f"Adjusted R-squared: {model.rsquared_adj:.4f}"),
                    html.P(f"p-value: {model.f_pvalue:.4f}"),
                    html.P(f"Coefficient: {model.params[1]:.4f}")
                ])
    except Exception as e:
        print(f"Error in regression analysis: {e}")
        regression_output = html.Div([
            html.H5("Regression Error"),
            html.P(f"An error occurred: {str(e)}")
        ])
    
    # Prepare data for the table
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
        
        # Format the columns for the data table
        formatted_columns = []
        for col in table_columns:
            if col == 'Date':
                formatted_columns.append({"name": "Date", "id": "Date", "type": "datetime"})
            elif col in ['Δ Revenue', 'Δ 30 day', 'OCR', '30 day bill', 'Lag Revenue Change']:
                formatted_columns.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ".2%"}})
            else:
                formatted_columns.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ",.2f"}})
    except Exception as e:
        print(f"Error preparing data table: {e}")
        # Provide fallback empty table
        table_data = []
        formatted_columns = [{"name": "Error", "id": "error"}]
    
    return fig_time_series, fig_correlation, fig_distribution, regression_output, table_data, formatted_columns


if __name__ == '__main__':
    app.run_server(debug=True)