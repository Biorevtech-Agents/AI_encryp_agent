"""
Data utility functions for the trading agent.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

def preprocess_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess market data by cleaning and formatting.
    """
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.fillna(method='ffill')
    return df

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from price series.
    """
    return prices.pct_change()

def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def normalize_data(data: pd.Series) -> pd.Series:
    """
    Normalize data using min-max scaling.
    """
    return (data - data.min()) / (data.max() - data.min())

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical analysis features.
    """
    df = data.copy()
    df['returns'] = calculate_returns(df['close'])
    df['volatility'] = calculate_volatility(df['returns'])
    df['normalized_volume'] = normalize_data(df['volume'])
    return df

def split_data(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def resample_data(data: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
    """
    Resample data to a different timeframe.
    """
    ohlcv = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return data.resample(timeframe).agg(ohlcv)

def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    stop_loss_pct: float
) -> float:
    """
    Calculate position size based on risk parameters.
    """
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size

def calculate_portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    """
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def format_trade_data(
    trades: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Format trade data into a DataFrame.
    """
    df = pd.DataFrame(trades)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    return df 