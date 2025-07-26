import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt, timedelta
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
        template='plotly_dark',
        paper_bgcolor='#34495e',
        plot_bgcolor='#2c3e50',
        font=dict(color='white'),
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
    fig.add_trace(go.Scatter(x=df[date_col], y=df['EWA_20'], mode='lines+markers', name='EMA 20', line=dict(color='#3498db'), marker=dict(size=4)))

    fig.update_layout(
        title="Exponential Moving Average vs Date",
        template='plotly_dark',
        paper_bgcolor='#34495e',
        plot_bgcolor='#2c3e50',
        font=dict(color='white'),
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
app.layout = html.Div(className="container", style={'backgroundColor': '#2c3e50', 'minHeight': '100vh'}, children=[
    # --- Navigation/Input Section ---
    html.Div(
        className="nav",
        style={
            'backgroundColor': '#34495e',
            'padding': '25px',
            'borderRadius': '12px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
            'margin': '20px',
            'width': '350px',
            'position': 'fixed',
            'height': 'calc(100vh - 40px)',
            'overflowY': 'auto'
        },
        children=[
            html.H1("Stock Dash App", className="start", style={'color': '#3498db', 'marginBottom': '30px', 'fontSize': '28px'}),
            
            # Stock Code Input Section
            html.Div(className="input-container", style={'marginBottom': '25px'}, children=[
                html.H3("Input stock code:", style={'marginBottom': '12px', 'color': '#ecf0f1', 'fontSize': '16px', 'fontWeight': 'bold'}),
                html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[
                    dcc.Input(
                        id='stock_code_input', 
                        type='text', 
                        placeholder='e.g., GOOGL', 
                        value='GOOGL',
                        style={
                            'width': '180px', 
                            'padding': '12px', 
                            'border': '2px solid #e8ecf0',
                            'borderRadius': '8px',
                            'fontSize': '14px',
                            'outline': 'none',
                            'transition': 'border-color 0.3s'
                        }
                    ),
                    html.Button(
                        'Submit', 
                        id='submit_stock_code_button', 
                        n_clicks=0, 
                        style={
                            'backgroundColor': '#27ae60', 
                            'border': 'none', 
                            'padding': '12px 18px',
                            'color': 'white',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'transition': 'background-color 0.3s'
                        }
                    )
                ])
            ]),
            
            # Improved Date Input Section
            html.Div(className="input-container", style={'marginBottom': '25px'}, children=[
                html.H3("Select time period:", style={'marginBottom': '15px', 'color': '#ecf0f1', 'fontSize': '16px', 'fontWeight': 'bold'}),
                
                # Quick Date Presets - Improved styling
                html.Div(style={'marginBottom': '20px'}, children=[
                    html.Label("Quick Presets:", style={'display': 'block', 'marginBottom': '10px', 'fontWeight': 'bold', 'color': '#ecf0f1', 'fontSize': '14px'}),
                    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'}, children=[
                        html.Button('1 Year', id='preset_1y', n_clicks=0, 
                                  style={'padding': '10px 12px', 'backgroundColor': '#3498db', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 
                                        'fontSize': '12px', 'fontWeight': 'bold', 'transition': 'background-color 0.3s'}),
                        html.Button('6 Months', id='preset_6m', n_clicks=0, 
                                  style={'padding': '10px 12px', 'backgroundColor': '#3498db', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 
                                        'fontSize': '12px', 'fontWeight': 'bold', 'transition': 'background-color 0.3s'}),
                        html.Button('3 Months', id='preset_3m', n_clicks=0, 
                                  style={'padding': '10px 12px', 'backgroundColor': '#3498db', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 
                                        'fontSize': '12px', 'fontWeight': 'bold', 'transition': 'background-color 0.3s'}),
                        html.Button('1 Month', id='preset_1m', n_clicks=0, 
                                  style={'padding': '10px 12px', 'backgroundColor': '#3498db', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 
                                        'fontSize': '12px', 'fontWeight': 'bold', 'transition': 'background-color 0.3s'}),
                    ])
                ]),
                
                # Manual Date Input - Improved styling
                html.Div(style={'backgroundColor': "#325e89", 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '15px', 'border': '1px solid #4a5f7a'}, children=[
                    html.Label("Custom Date Range:", style={'display': 'block', 'marginBottom': '12px', 'fontWeight': 'bold', 'color': '#ecf0f1', 'fontSize': '14px'}),
                    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px'}, children=[
                        html.Div(children=[
                            html.Label("Start Date:", style={'display': 'block', 'marginBottom': '6px', 'color': '#ecf0f1', 'fontSize': '13px', 'fontWeight': 'bold'}),
                            dcc.Input(
                                id='start_date_input',
                                type='text',
                                placeholder='YYYY-MM-DD',
                                value='2020-01-01',
                                style={
                                    'width': '100%', 
                                    'padding': '10px', 
                                    'border': '2px solid #dee2e6', 
                                    'borderRadius': '6px',
                                    'fontSize': '13px',
                                    'backgroundColor': 'white',
                                    'color' : '#000000'
                                }
                            )
                        ]),
                        html.Div(children=[
                            html.Label("End Date:", style={'display': 'block', 'marginBottom': '6px', 'color': '#ecf0f1', 'fontSize': '13px', 'fontWeight': 'bold'}),
                            dcc.Input(
                                id='end_date_input',
                                type='text',
                                placeholder='YYYY-MM-DD',
                                value=dt.now().strftime('%Y-%m-%d'),
                                style={
                                    'width': '100%', 
                                    'padding': '10px', 
                                    'border': '2px solid #dee2e6', 
                                    'borderRadius': '6px',
                                    'fontSize': '13px',
                                    'backgroundColor': 'white',
                                    'color' : '#000000'
                                }
                            )
                        ])
                    ])
                ]),
                
                # Enhanced Date Picker
                html.Details([
                    html.Summary("📅 Use Visual Date Picker", 
                               style={'cursor': 'pointer', 'color': '#3498db', 'fontWeight': 'bold', 
                                     'padding': '10px', 'marginBottom': '10px', 'fontSize': '14px',
                                     'backgroundColor': '#2c3e50', 'borderRadius': '6px'}),
                    html.Div(style={'marginTop': '10px', 'padding': '15px', 'backgroundColor': '#ffffff', 
                                   'border': '2px solid #3498db', 'borderRadius': '8px'}, children=[
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
                ]),
                
                # Date validation message
                html.Div(id='date_validation_message', 
                        style={'color': '#e74c3c', 'fontSize': '12px', 'marginTop': '8px', 'fontWeight': 'bold'})
            ]),
            
            # Analysis Buttons - Same size styling
            html.Div(className="input-container", style={'marginBottom': '25px'}, children=[
                html.H3("Analysis Options:", style={'marginBottom': '15px', 'color': '#ecf0f1', 'fontSize': '16px', 'fontWeight': 'bold'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '10px', 'marginBottom': '15px'}, children=[
                    html.Button(
                        'Stock Price', 
                        id='stock_price_button', 
                        n_clicks=0, 
                        style={
                            'backgroundColor': '#e67e22', 
                            'border': 'none', 
                            'padding': '14px 20px',
                            'color': 'white',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'width': '100%',
                            'height': '48px',
                            'transition': 'background-color 0.3s'
                        }
                    ),
                    html.Button(
                        'Indicators', 
                        id='indicators_button', 
                        n_clicks=0, 
                        style={
                            'backgroundColor': '#8e44ad', 
                            'border': 'none', 
                            'padding': '14px 20px',
                            'color': 'white',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'width': '100%',
                            'height': '48px',
                            'transition': 'background-color 0.3s'
                        }
                    ),
                ]),
            ]),
            
            # Forecast Section
            html.Div(className="input-container", children=[
                html.H3("Forecast:", style={'marginBottom': '15px', 'color': '#ecf0f1', 'fontSize': '16px', 'fontWeight': 'bold'}),
                html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
                    dcc.Input(
                        id='forecast_days_input', 
                        type='number', 
                        placeholder='Days (1-30)', 
                        min=1, 
                        max=30,
                        style={
                            'width': '100%', 
                            'padding': '12px', 
                            'border': '2px solid #e8ecf0', 
                            'borderRadius': '8px',
                            'fontSize': '14px',
                            'outline': 'none'
                        }
                    ),
                    html.Button(
                        'Generate Forecast', 
                        id='forecast_button', 
                        n_clicks=0, 
                        style={
                            'backgroundColor': '#27ae60', 
                            'border': 'none', 
                            'padding': '14px 20px',
                            'color': 'white',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'width': '100%',
                            'transition': 'background-color 0.3s'
                        }
                    )
                ])
            ]),
        ]
    ),
    # --- Content/Output Section - Updated with blue theme ---
    html.Div(
        className="content",
        style={
            'marginLeft': '390px', 
            'padding': '20px',
            'backgroundColor': '#34495e',  # Changed from '#ecf0f1' to match sidebar
            'margin': '20px 20px 20px 390px',
            'borderRadius': '12px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
            'minHeight': 'calc(100vh - 40px)'
        },
        children=[
            html.Div(id="header", className="header"),
            html.Div(id="description", className="description_ticker", style={'color': '#ecf0f1'}),  # Added white text color
            html.Div(id="graphs-content"),
            html.Div(id="main-content"),
            html.Div(id="forecast-content")
        ]
    )
])

# --- Callbacks ---

# NEW: Callback to clear previous results when submit button is clicked
@app.callback(
    [Output("graphs-content", "children", allow_duplicate=True),
     Output("main-content", "children", allow_duplicate=True),
     Output("forecast-content", "children", allow_duplicate=True)],
    [Input("submit_stock_code_button", "n_clicks")],
    prevent_initial_call=True
)
def clear_previous_results(n_clicks):
    """Clear all previous graphs when submit button is clicked"""
    if n_clicks > 0:
        return [], [], []
    raise PreventUpdate

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
        start_date = end_date - timedelta(days=365)
    elif button_id == 'preset_6m':
        start_date = end_date - timedelta(days=180)
    elif button_id == 'preset_3m':
        start_date = end_date - timedelta(days=90)
    elif button_id == 'preset_1m':
        start_date = end_date - timedelta(days=30)
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
        return "⚠️ Please enter both start and end dates"
    
    start_date = validate_date_string(start_date_str)
    end_date = validate_date_string(end_date_str)
    
    if not start_date:
        return "❌ Invalid start date format. Use YYYY-MM-DD"
    if not end_date:
        return "❌ Invalid end date format. Use YYYY-MM-DD"
    if start_date >= dt.now().date():
        return "❌ Start date must be in the past"
    if end_date > dt.now().date():
        return "❌ End date cannot be in the future"
    if start_date >= end_date:
        return "❌ Start date must be before end date"
    
    return "✅ Date range is valid"

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
        html.H1(f"{short_name}.", style={'color': '#ecf0f1'})  # Added white text color
    ]
    return summary, header_children

# Callback to update stock price graph
@app.callback(
    Output("graphs-content", "children", allow_duplicate=True),
    [Input("stock_price_button", "n_clicks")],
    [State("stock_code_input", "value"), State("start_date_input", "value"), State("end_date_input", "value")],
    prevent_initial_call=True
)
def update_stock_graph(n_clicks, stock_code, start_date_str, end_date_str):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    
    # Validate dates first
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    
    if not start_date or not end_date:
        return html.P("Please enter valid dates in YYYY-MM-DD format", style={'color': '#e74c3c'})
    
    if start_date >= end_date:
        return html.P("Start date must be before end date", style={'color': '#e74c3c'})
    
    try:
        # Download data
        data = yf.download(stock_code, start=start_date, end=end_date)
        
        if data.empty:
            return html.P(f"No data found for {stock_code} in the selected date range.", style={'color': '#f39c12'})
        
        # Reset index to get Date as a column
        df = data.reset_index()
        
        return dcc.Graph(figure=get_stock_price_fig(df))
    except Exception as e:
        return html.P(f"An error occurred: {e}", style={'color': '#e74c3c'})

# Callback to update indicator graph
@app.callback(
    Output("main-content", "children", allow_duplicate=True),
    [Input("indicators_button", "n_clicks")],
    [State("stock_code_input", "value"), State("start_date_input", "value"), State("end_date_input", "value")],
    prevent_initial_call=True
)
def update_indicator_graph(n_clicks, stock_code, start_date_str, end_date_str):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    
    # Validate dates first
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    
    if not start_date or not end_date:
        return html.P("Please enter valid dates in YYYY-MM-DD format", style={'color': '#e74c3c'})
    
    if start_date >= end_date:
        return html.P("Start date must be before end date", style={'color': '#e74c3c'})
    
    try:
        # Download data
        data = yf.download(stock_code, start=start_date, end=end_date)
        if data.empty:
            return html.P(f"No data found for {stock_code} in the selected date range.", style={'color': '#f39c12'})
        
        df = data.reset_index()
        return dcc.Graph(figure=get_more_fig(df))
    except Exception as e:
        return html.P(f"An error occurred: {e}", style={'color': '#e74c3c'})

# Callback to update forecast graph
@app.callback(
    Output("forecast-content", "children", allow_duplicate=True),
    [Input("forecast_button", "n_clicks")],
    [State("stock_code_input", "value"), State("forecast_days_input", "value")],
    prevent_initial_call=True
)
def update_forecast_graph(n_clicks, stock_code, n_days):
    if n_clicks == 0 or not n_days or not stock_code:
        raise PreventUpdate

    try:
        # Limit forecast to maximum 30 days
        if n_days > 30:
            return html.P("Forecast is only available for the next 30 days. Please enter a value between 1 and 30.", style={'color': '#f39c12'})
        
        if n_days < 1:
            return html.P("Please enter a positive number of days for forecasting.", style={'color': '#f39c12'})
        
        # Get the forecast
        forecast_df = predict_stock_price(stock_code, int(n_days))
        
        if forecast_df is None:
            return html.P("Could not generate forecast. Please check the stock ticker.", style={'color': '#f39c12'})

        # Create the figure to match the requested style
        fig = go.Figure()
        
        # Add forecast data trace
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], 
            y=forecast_df['Prediction'], 
            mode='lines+markers', 
            name='Predicted Close Price',
            line=dict(color='#3498db'),
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
            arrowcolor="#3498db",
            ax=20,
            ay=-30,
            bgcolor="#3498db",
            bordercolor="#3498db",
            borderwidth=2,
            font=dict(color="white")
        )
        
        # Update layout for a clean, focused look with dark theme
        fig.update_layout(
            title=f"Predicted Close Price of next {n_days} days",
            template='plotly_dark',
            paper_bgcolor='#34495e',
            plot_bgcolor='#2c3e50',
            font=dict(color='white'),
            xaxis_title="Date",
            yaxis_title="Close Price",
            showlegend=False
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"An error occurred during forecast: {e}", style={'color': '#e74c3c'})

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)
    # .\venv\Scripts\activate