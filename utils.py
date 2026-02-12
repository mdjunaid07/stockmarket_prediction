# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def download_data(ticker: str, start=None, end=None):
    """
    Download historical data from yfinance and return a cleaned DataFrame.
    """
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df

def add_technical_indicators(df: pd.DataFrame):
    """
    Add simple indicators: MA7, MA21, Momentum, Volatility
    """
    df = df.copy()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(7)
    df['Volatility'] = df['Close'].rolling(21).std()
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def prepare_features(df: pd.DataFrame, days_ahead=7):
    """
    Create X, y where target is close shifted by -days_ahead.
    Returns X (2D array), y (1D array), and feature column names.
    """
    df = df.copy()
    df = add_technical_indicators(df)
    df['Target'] = df['Close'].shift(-days_ahead)
    df = df.dropna()
    feature_cols = ['Close', 'MA7', 'MA21', 'Momentum', 'Volatility', 'Volume']
    X = df[feature_cols].values
    y = df['Target'].values
    return X, y, feature_cols, df

def build_and_train_model(X_train, y_train):
    """
    Build a simple pipeline (scaler + LinearRegression). Returns trained pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(model, path='models/linear_model.joblib'):
    dump(model, path)

def load_model(path='models/linear_model.joblib'):
    return load(path)

def iterative_predict(last_row: pd.Series, model, feature_cols, days_ahead=7):
    """
    last_row: pd.Series of the most recent features (with feature_cols present)
    model: trained pipeline
    feature_cols: order of features used by model
    returns: array of predictions for next days (iterative)
    """
    preds = []
    cur_features = last_row[feature_cols].copy()  # pandas Series
    cur_features = cur_features.astype(float)

    for i in range(days_ahead):
        x = cur_features.values.reshape(1, -1)
        p = model.predict(x)[0]
        preds.append(p)
        # update features for next iteration:
        # shift Close to the predicted value, update moving averages/momentum roughly
        # simple approach:
        cur_features['Close'] = p
        cur_features['MA7'] = (cur_features['MA7'] * 6 + p) / 7
        cur_features['MA21'] = (cur_features['MA21'] * 20 + p) / 21
        cur_features['Momentum'] = p - cur_features['Close']  # approximate
        # Volume keep same (or set to recent volume)
    return np.array(preds)
