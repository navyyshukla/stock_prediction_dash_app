import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def predict_stock_price(ticker, n_days):
    if n_days > 30:
        n_days = 30
    elif n_days < 1:
        return None

    today = date.today()
    start_date = today - timedelta(days=730)
    
    try:
        data = yf.download(ticker, start=start_date, end=today, progress=False)
        if data.empty:
            print(f"No data found for ticker: {ticker}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns.values]
        df = data.reset_index()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    try:
        for lag in [1, 3, 5, 7]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        df['Price_Change'] = df['Close'].pct_change()
        if 'Volume' in df.columns:
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
        df.dropna(inplace=True)
        if len(df) < 50:
            print("Not enough data to train the model after feature engineering.")
            return None
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return None

    exclude_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns].fillna(method='bfill').fillna(method='ffill')
    y = df['Close']

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        print(f"Error in feature scaling: {e}")
        return None

    try:
        svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)
        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6)
        ensemble_model = VotingRegressor(estimators=[('svr', svr), ('rf', rf), ('gb', gb)])
        ensemble_model.fit(X_scaled, y)
    except Exception as e:
        print(f"Error training ensemble model: {e}")
        return None

    try:
        last_features = X.iloc[-1:].copy()
        predictions = []
        recent_closes = list(y.tail(20).values)
        
        for i in range(n_days):
            last_features_scaled = scaler.transform(last_features.values)
            next_pred = ensemble_model.predict(last_features_scaled)[0]
            last_close = recent_closes[-1]
            if next_pred < last_close * 0.7:
                next_pred = last_close * 0.9
            elif next_pred > last_close * 1.5:
                next_pred = last_close * 1.1
            predictions.append(float(next_pred))
            recent_closes.append(next_pred)
            new_features = last_features.iloc[0].copy()
            if 'Close_lag_7' in new_features.index:
                new_features['Close_lag_7'] = new_features.get('Close_lag_5', next_pred)
            if 'Close_lag_5' in new_features.index:
                new_features['Close_lag_5'] = new_features.get('Close_lag_3', next_pred)
            if 'Close_lag_3' in new_features.index:
                new_features['Close_lag_3'] = new_features.get('Close_lag_1', next_pred)
            if 'Close_lag_1' in new_features.index:
                new_features['Close_lag_1'] = next_pred
            if len(recent_closes) >= 20:
                if 'MA_5' in new_features.index:
                    new_features['MA_5'] = np.mean(recent_closes[-5:])
                if 'MA_10' in new_features.index:
                    new_features['MA_10'] = np.mean(recent_closes[-10:])
                if 'MA_20' in new_features.index:
                    new_features['MA_20'] = np.mean(recent_closes[-20:])
                if 'Volatility_10' in new_features.index:
                    new_features['Volatility_10'] = np.std(recent_closes[-10:])
                if 'Momentum' in new_features.index:
                    new_features['Momentum'] = next_pred - recent_closes[-5]
                if 'Price_Change' in new_features.index:
                    new_features['Price_Change'] = (next_pred - recent_closes[-2]) / recent_closes[-2] if len(recent_closes) > 1 else 0
            if 'Volume_Ratio' in new_features.index:
                new_features['Volume_Ratio'] = 1.0
            last_features = pd.DataFrame([new_features.values], columns=feature_columns)
    except Exception as e:
        print(f"Error in prediction loop: {e}")
        return None

    try:
        last_real_date = df['Date'].iloc[-1]
        future_dates = []
        current_date = last_real_date
        days_added = 0
        while days_added < n_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
                days_added += 1
        if len(future_dates) != len(predictions):
            min_length = min(len(future_dates), len(predictions))
            future_dates = future_dates[:min_length]
            predictions = predictions[:min_length]
        forecast_df = pd.DataFrame({
            'Date': pd.to_datetime(future_dates),
            'Prediction': np.array(predictions, dtype=np.float64)
        })
        print(f"Successfully generated {len(predictions)}-day forecast for {ticker}")
        return forecast_df
    except Exception as e:
        print(f"Error creating forecast DataFrame: {e}")
        return None

def validate_prediction_inputs(ticker, n_days):
    if not isinstance(ticker, str) or not ticker.strip():
        print("Invalid ticker: must be a non-empty string")
        return False
    if not isinstance(n_days, (int, float)) or n_days < 1 or n_days > 30:
        print("Invalid n_days: must be between 1 and 30")
        return False
    return True
