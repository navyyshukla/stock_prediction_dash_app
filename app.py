import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Import the prediction model from model.py
from model import predict_stock_price

# --- App Instantiation ---
app = dash.Dash(__name__, assets_folder='assets')
server = app.server

# --- Helper Functions for Plotly Graphs ---
def get_stock_price_fig(df):
    """Generates the stock price graph."""
    fig = go.Figure()
    
    # Handle multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        # Get the first ticker symbol from multi-level columns
        ticker_col = df.columns.levels[1][0] if len(df.columns.levels) > 1 else None
        open_col = ('Open', ticker_col) if ticker_col else 'Open'
        close_col = ('Close', ticker_col) if ticker_col else 'Close'
        date_col = 'Date'
    else:
        # Single level columns
        open_col = 'Open'
        close_col = 'Close' 
        date_col = 'Date'
    
    # Add Open and Close price traces
    fig.add_trace(go.Scatter(
        x=df[date_col], 
        y=df[open_col], 
        mode='lines', 
        name='Open', 
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], 
        y=df[close_col], 
        mode='lines', 
        name='Close', 
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="Closing and Opening Price vs Date",
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(
            title="Price Type",
            x=1.02,
            y=1
        )
    )
    return fig

def get_more_fig(df):
    """Generates the EMA indicator graph."""
    # Handle multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        ticker_col = df.columns.levels[1][0] if len(df.columns.levels) > 1 else None
        close_col = ('Close', ticker_col) if ticker_col else 'Close'
        date_col = 'Date'
    else:
        close_col = 'Close'
        date_col = 'Date'
    
    df['EWA_20'] = df[close_col].ewm(span=20, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df['EWA_20'], mode='lines+markers', name='EMA 20', line=dict(color='blue'), marker=dict(size=4)))

    fig.update_layout(
        title="Exponential Moving Average vs Date",
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        xaxis_title="Date",
        yaxis_title="EWA_20"
    )
    return fig

def validate_date_string(date_string):
    """Validate and parse date string in YYYY-MM-DD format"""
    try:
        return dt.strptime(date_string, '%Y-%m-%d').date()
    except ValueError:
        return None

# --- App Layout ---
app.layout = html.Div(className="container", children=[
    # --- Navigation/Input Section ---
    html.Div(
        className="nav",
        children=[
            html.H1("Stock Dash App", className="start"),
            html.Div(className="input-container", children=[
                html.H3("Input stock code:"),
                dcc.Input(
                    id='stock_code_input', 
                    type='text', 
                    placeholder='e.g., GOOGL', 
                    value='GOOGL',
                    style={'width': '200px', 'padding': '5px', 'margin-bottom': '10px'}
                ),
                html.Button(
                    'Submit', 
                    id='submit_stock_code_button', 
                    n_clicks=0, 
                    style={
                        'background-color': '#00BFFF', 
                        'border': 'none', 
                        'padding': '8px 15px',
                        'color': 'white',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                )
            ]),
            
            # Improved Date Input Section
            html.Div(className="input-container", children=[
                html.H3("Select start and end dates:"),
                html.Div(style={'display': 'flex', 'flex-direction': 'column', 'gap': '10px'}, children=[
                    # Manual Date Input Option
                    html.Div(style={'display': 'flex', 'gap': '10px', 'align-items': 'center'}, children=[
                        html.Label("Start Date:", style={'min-width': '80px'}),
                        dcc.Input(
                            id='start_date_input',
                            type='text',
                            placeholder='YYYY-MM-DD (e.g., 2020-01-01)',
                            value='2020-01-01',
                            style={'width': '200px', 'padding': '5px', 'border': '1px solid #ccc', 'border-radius': '4px'}
                        )
                    ]),
                    html.Div(style={'display': 'flex', 'gap': '10px', 'align-items': 'center'}, children=[
                        html.Label("End Date:", style={'min-width': '80px'}),
                        dcc.Input(
                            id='end_date_input',
                            type='text',
                            placeholder='YYYY-MM-DD (e.g., 2024-12-31)',
                            value=dt.now().strftime('%Y-%m-%d'),
                            style={'width': '200px', 'padding': '5px', 'border': '1px solid #ccc', 'border-radius': '4px'}
                        )
                    ]),
                    
                    # Quick Date Presets
                    html.Div(style={'margin-top': '10px'}, children=[
                        html.Label("Quick Presets:", style={'margin-right': '10px'}),
                        html.Button('Last 1 Year', id='preset_1y', n_clicks=0, style={'margin-right': '5px', 'padding': '4px 8px', 'font-size': '12px'}),
                        html.Button('Last 6 Months', id='preset_6m', n_clicks=0, style={'margin-right': '5px', 'padding': '4px 8px', 'font-size': '12px'}),
                        html.Button('Last 3 Months', id='preset_3m', n_clicks=0, style={'margin-right': '5px', 'padding': '4px 8px', 'font-size': '12px'}),
                        html.Button('Last Month', id='preset_1m', n_clicks=0, style={'padding': '4px 8px', 'font-size': '12px'}),
                    ]),
                    
                    # Fallback Date Picker (better styled)
                    html.Details([
                        html.Summary("Or use date picker", style={'cursor': 'pointer', 'color': '#666', 'margin-top': '10px'}),
                        html.Div(style={'margin-top': '10px', 'padding': '10px', 'border': '1px solid #eee', 'border-radius': '4px'}, children=[
                            dcc.DatePickerRange(
                                id='date_picker_range',
                                min_date_allowed=dt(1970, 1, 1),
                                max_date_allowed=dt.now(),
                                start_date=dt(2020, 1, 1),
                                end_date=dt.now(),
                                display_format='YYYY-MM-DD',
                                style={'width': '100%'},
                                calendar_orientation='horizontal'
                            )
                        ])
                    ])
                ]),
                
                # Date validation message
                html.Div(id='date_validation_message', style={'color': 'red', 'font-size': '12px', 'margin-top': '5px'})
            ]),
            
            html.Div(className="input-container", children=[
                html.Button(
                    'Stock Price', 
                    id='stock_price_button', 
                    n_clicks=0, 
                    style={
                        'background-color': '#00BFFF', 
                        'border': 'none', 
                        'padding': '8px 15px', 
                        'margin-right': '10px',
                        'color': 'white',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                ),
                html.Button(
                    'Indicators', 
                    id='indicators_button', 
                    n_clicks=0, 
                    style={
                        'background-color': '#00BFFF', 
                        'border': 'none', 
                        'padding': '8px 15px',
                        'color': 'white',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                ),
            ]),
            html.Div(className="input-container", children=[
                dcc.Input(
                    id='forecast_days_input', 
                    type='number', 
                    placeholder='Number of days (1-30)', 
                    min=1, 
                    max=30,
                    style={'margin-bottom': '10px', 'width': '200px', 'padding': '5px', 'border': '1px solid #ccc', 'border-radius': '4px'}
                ),
                html.Button(
                    'Forecast', 
                    id='forecast_button', 
                    n_clicks=0, 
                    style={
                        'background-color': '#90EE90', 
                        'border': 'none', 
                        'padding': '8px 15px',
                        'color': 'black',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                )
            ]),
        ]
    ),
    # --- Content/Output Section ---
    html.Div(
        className="content",
        children=[
            html.Div(id="header", className="header"),
            html.Div(id="description", className="description_ticker"),
            html.Div(id="graphs-content"),
            html.Div(id="main-content"),
            html.Div(id="forecast-content")
        ]
    )
])

# --- Callbacks ---

# Callback for date presets
@app.callback(
    [Output('start_date_input', 'value'), Output('end_date_input', 'value')],
    [Input('preset_1y', 'n_clicks'), Input('preset_6m', 'n_clicks'), 
     Input('preset_3m', 'n_clicks'), Input('preset_1m', 'n_clicks')]
)
def update_date_presets(n1y, n6m, n3m, n1m):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    end_date = dt.now()
    
    if button_id == 'preset_1y':
        start_date = dt(end_date.year - 1, end_date.month, end_date.day)
    elif button_id == 'preset_6m':
        start_date = dt(end_date.year, max(1, end_date.month - 6), end_date.day)
        if end_date.month <= 6:
            start_date = dt(end_date.year - 1, end_date.month + 6, end_date.day)
    elif button_id == 'preset_3m':
        start_date = dt(end_date.year, max(1, end_date.month - 3), end_date.day)
        if end_date.month <= 3:
            start_date = dt(end_date.year - 1, end_date.month + 9, end_date.day)
    elif button_id == 'preset_1m':
        start_date = dt(end_date.year, max(1, end_date.month - 1), end_date.day)
        if end_date.month == 1:
            start_date = dt(end_date.year - 1, 12, end_date.day)
    else:
        raise PreventUpdate
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Callback to sync date picker with manual inputs
@app.callback(
    [Output('date_picker_range', 'start_date'), Output('date_picker_range', 'end_date')],
    [Input('start_date_input', 'value'), Input('end_date_input', 'value')]
)
def sync_date_picker(start_date_str, end_date_str):
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    
    return start_date, end_date

# Callback to validate dates
@app.callback(
    Output('date_validation_message', 'children'),
    [Input('start_date_input', 'value'), Input('end_date_input', 'value')]
)
def validate_dates(start_date_str, end_date_str):
    if not start_date_str or not end_date_str:
        return "Please enter both start and end dates"
    
    start_date = validate_date_string(start_date_str)
    end_date = validate_date_string(end_date_str)
    
    if not start_date:
        return "Invalid start date format. Use YYYY-MM-DD"
    if not end_date:
        return "Invalid end date format. Use YYYY-MM-DD"
    if start_date >= dt.now().date():
        return "Start date must be in the past"
    if end_date > dt.now().date():
        return "End date cannot be in the future"
    if start_date >= end_date:
        return "Start date must be before end date"
    
    return ""

# Callback to update company information
@app.callback(
    [Output("description", "children"), Output("header", "children")],
    [Input("submit_stock_code_button", "n_clicks")],
    [State("stock_code_input", "value")]
)
def update_company_info(n_clicks, stock_code):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    
    ticker = yf.Ticker(stock_code)
    try:
        inf = ticker.info
        if len(inf) <= 1:
            return f"Could not retrieve any information for ticker: {stock_code}. It may be invalid.", ""
            
    except Exception as e:
        return f"Error fetching data for {stock_code}. Please check the ticker symbol.", ""

    summary = inf.get('longBusinessSummary', 'No detailed business summary available.')
    logo_url = inf.get('logo_url', '')
    short_name = inf.get('shortName', stock_code)

    header_children = [
        html.Img(src=logo_url, style={'height': '80px', 'width': '80px', 'display': 'block' if logo_url else 'none'}),
        html.H1(f"{short_name}.")
    ]
    return summary, header_children

# Callback to update stock price graph
@app.callback(
    Output("graphs-content", "children"),
    [Input("stock_price_button", "n_clicks")],
    [State("stock_code_input", "value"), State("start_date_input", "value"), State("end_date_input", "value")]
)
def update_stock_graph(n_clicks, stock_code, start_date_str, end_date_str):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    
    # Validate dates first
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    
    if not start_date or not end_date:
        return html.P("Please enter valid dates in YYYY-MM-DD format", style={'color': 'red'})
    
    if start_date >= end_date:
        return html.P("Start date must be before end date", style={'color': 'red'})
    
    try:
        # Download data
        data = yf.download(stock_code, start=start_date, end=end_date)
        
        if data.empty:
            return html.P(f"No data found for {stock_code} in the selected date range.", style={'color': 'orange'})
        
        # Reset index to get Date as a column
        df = data.reset_index()
        
        return dcc.Graph(figure=get_stock_price_fig(df))
    except Exception as e:
        return html.P(f"An error occurred: {e}", style={'color': 'red'})

# Callback to update indicator graph
@app.callback(
    Output("main-content", "children"),
    [Input("indicators_button", "n_clicks")],
    [State("stock_code_input", "value"), State("start_date_input", "value"), State("end_date_input", "value")]
)
def update_indicator_graph(n_clicks, stock_code, start_date_str, end_date_str):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    
    # Validate dates first
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    
    if not start_date or not end_date:
        return html.P("Please enter valid dates in YYYY-MM-DD format", style={'color': 'red'})
    
    if start_date >= end_date:
        return html.P("Start date must be before end date", style={'color': 'red'})
    
    try:
        # Download data
        data = yf.download(stock_code, start=start_date, end=end_date)
        if data.empty:
            return html.P(f"No data found for {stock_code} in the selected date range.", style={'color': 'orange'})
        
        df = data.reset_index()
        return dcc.Graph(figure=get_more_fig(df))
    except Exception as e:
        return html.P(f"An error occurred: {e}", style={'color': 'red'})

# Callback to update forecast graph
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast_button", "n_clicks")],
    [State("stock_code_input", "value"), State("forecast_days_input", "value")]
)
def update_forecast_graph(n_clicks, stock_code, n_days):
    if n_clicks == 0 or not n_days or not stock_code:
        raise PreventUpdate

    try:
        # Limit forecast to maximum 30 days
        if n_days > 30:
            return html.P("Forecast is only available for the next 30 days. Please enter a value between 1 and 30.", style={'color': 'orange'})
        
        if n_days < 1:
            return html.P("Please enter a positive number of days for forecasting.", style={'color': 'orange'})
        
        # Get the forecast
        forecast_df = predict_stock_price(stock_code, int(n_days))
        
        if forecast_df is None:
            return html.P("Could not generate forecast. Please check the stock ticker.", style={'color': 'orange'})

        # Create the figure to match the requested style
        fig = go.Figure()
        
        # Add forecast data trace
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], 
            y=forecast_df['Prediction'], 
            mode='lines+markers', 
            name='Predicted Close Price',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))
        
        # Add hover text showing the last point value
        last_date = forecast_df['Date'].iloc[-1]
        last_value = forecast_df['Prediction'].iloc[-1]
        
        fig.add_annotation(
            x=last_date,
            y=last_value,
            text=f"({last_date.strftime('%b %d, %Y')}, {last_value:.3f})",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue",
            ax=20,
            ay=-30,
            bgcolor="blue",
            bordercolor="blue",
            borderwidth=2,
            font=dict(color="white")
        )
        
        # Update layout for a clean, focused look
        fig.update_layout(
            title=f"Predicted Close Price of next {n_days} days",
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            xaxis_title="Date",
            yaxis_title="Close Price",
            showlegend=False
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"An error occurred during forecast: {e}", style={'color': 'red'})

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)
    # .\venv\Scripts\activate