import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt, timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from model import predict_stock_price

app = dash.Dash(__name__, assets_folder='assets')
server = app.server

def get_stock_price_fig(df):
    fig = go.Figure()
    if isinstance(df.columns, pd.MultiIndex):
        ticker_col = df.columns.levels[1][0] if len(df.columns.levels) > 1 else None
        open_col = ('Open', ticker_col) if ticker_col else 'Open'
        close_col = ('Close', ticker_col) if ticker_col else 'Close'
        date_col = 'Date'
    else:
        open_col = 'Open'
        close_col = 'Close' 
        date_col = 'Date'
    fig.add_trace(go.Scatter(x=df[date_col], y=df[open_col], mode='lines', name='Open', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[close_col], mode='lines', name='Close', line=dict(color='blue')))
    fig.update_layout(
        title="Closing and Opening Price vs Date",
        template='plotly_dark',
        paper_bgcolor='#34495e',
        plot_bgcolor='#2c3e50',
        font=dict(color='white'),
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(title="Price Type", x=1.02, y=1)
    )
    return fig

def get_more_fig(df):
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
    try:
        return dt.strptime(date_string, '%Y-%m-%d').date()
    except ValueError:
        return None

app.layout = html.Div(className="container", style={'backgroundColor': '#2c3e50', 'minHeight': '100vh'}, children=[
    html.Div(className="nav", style={
        'backgroundColor': '#34495e',
        'padding': '25px',
        'borderRadius': '12px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
        'margin': '20px',
        'width': '350px',
        'position': 'fixed',
        'height': 'calc(100vh - 40px)',
        'overflowY': 'auto'
    }, children=[]),  # content omitted here for brevity

    html.Div(className="content", style={
        'marginLeft': '390px', 
        'padding': '20px',
        'backgroundColor': '#34495e',
        'margin': '20px 20px 20px 390px',
        'borderRadius': '12px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
        'minHeight': 'calc(100vh - 40px)'
    }, children=[
        html.Div(id="header", className="header"),
        html.Div(id="description", className="description_ticker", style={'color': '#ecf0f1'}),
        html.Div(id="graphs-content"),
        html.Div(id="main-content"),
        html.Div(id="forecast-content")
    ])
])

@app.callback(
    [Output("graphs-content", "children", allow_duplicate=True),
     Output("main-content", "children", allow_duplicate=True),
     Output("forecast-content", "children", allow_duplicate=True)],
    [Input("submit_stock_code_button", "n_clicks")],
    prevent_initial_call=True
)
def clear_previous_results(n_clicks):
    if n_clicks > 0:
        return [], [], []
    raise PreventUpdate

@app.callback(
    [Output('start_date_input', 'value'), Output('end_date_input', 'value')],
    [Input('preset_1y', 'n_clicks'), Input('preset_6m', 'n_clicks'), Input('preset_3m', 'n_clicks'), Input('preset_1m', 'n_clicks')]
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

@app.callback(
    [Output('date_picker_range', 'start_date'), Output('date_picker_range', 'end_date')],
    [Input('start_date_input', 'value'), Input('end_date_input', 'value')]
)
def sync_date_picker(start_date_str, end_date_str):
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    return start_date, end_date

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
        html.H1(f"{short_name}.", style={'color': '#ecf0f1'})
    ]
    return summary, header_children

@app.callback(
    Output("graphs-content", "children", allow_duplicate=True),
    [Input("stock_price_button", "n_clicks")],
    [State("stock_code_input", "value"), State("start_date_input", "value"), State("end_date_input", "value")],
    prevent_initial_call=True
)
def update_stock_graph(n_clicks, stock_code, start_date_str, end_date_str):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    if not start_date or not end_date:
        return html.P("Please enter valid dates in YYYY-MM-DD format", style={'color': '#e74c3c'})
    if start_date >= end_date:
        return html.P("Start date must be before end date", style={'color': '#e74c3c'})
    try:
        data = yf.download(stock_code, start=start_date, end=end_date)
        if data.empty:
            return html.P(f"No data found for {stock_code} in the selected date range.", style={'color': '#f39c12'})
        df = data.reset_index()
        return dcc.Graph(figure=get_stock_price_fig(df))
    except Exception as e:
        return html.P(f"An error occurred: {e}", style={'color': '#e74c3c'})

@app.callback(
    Output("main-content", "children", allow_duplicate=True),
    [Input("indicators_button", "n_clicks")],
    [State("stock_code_input", "value"), State("start_date_input", "value"), State("end_date_input", "value")],
    prevent_initial_call=True
)
def update_indicator_graph(n_clicks, stock_code, start_date_str, end_date_str):
    if n_clicks == 0 or not stock_code:
        raise PreventUpdate
    start_date = validate_date_string(start_date_str) if start_date_str else dt(2020, 1, 1).date()
    end_date = validate_date_string(end_date_str) if end_date_str else dt.now().date()
    if not start_date or not end_date:
        return html.P("Please enter valid dates in YYYY-MM-DD format", style={'color': '#e74c3c'})
    if start_date >= end_date:
        return html.P("Start date must be before end date", style={'color': '#e74c3c'})
    try:
        data = yf.download(stock_code, start=start_date, end=end_date)
        if data.empty:
            return html.P(f"No data found for {stock_code} in the selected date range.", style={'color': '#f39c12'})
        df = data.reset_index()
        return dcc.Graph(figure=get_more_fig(df))
    except Exception as e:
        return html.P(f"An error occurred: {e}", style={'color': '#e74c3c'})

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
        if n_days > 30:
            return html.P("Forecast is only available for the next 30 days. Please enter a value between 1 and 30.", style={'color': '#f39c12'})
        if n_days < 1:
            return html.P("Please enter a positive number of days for forecasting.", style={'color': '#f39c12'})
        forecast_df = predict_stock_price(stock_code, int(n_days))
        if forecast_df is None:
            return html.P("Could not generate forecast. Please check the stock ticker.", style={'color': '#f39c12'})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Prediction'], mode='lines+markers', name='Predicted Close Price', line=dict(color='#3498db'), marker=dict(size=6)))
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

if __name__ == '__main__':
    app.run(debug=True)
