"""
Tests for utility functions.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from src.utils.data_utils import (
    normalize_data,
    calculate_returns,
    calculate_volatility,
    preprocess_market_data,
    create_features,
    split_data,
    resample_data,
    calculate_position_size,
    calculate_portfolio_metrics,
    format_trade_data
)
from src.utils.technical_indicators import TechnicalIndicators
from src.utils.sentiment_utils import (
    preprocess_text,
    calculate_sentiment_score,
    aggregate_sentiment
)

# Test Data Utilities
def test_technical_indicators():
    """Test technical indicator calculations"""
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)
    
    # Add technical indicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Check if indicators are present
    assert 'rsi' in df.columns
    assert 'macd_line' in df.columns
    assert 'macd_signal' in df.columns
    assert 'bb_upper' in df.columns
    assert 'bb_lower' in df.columns
    assert 'ma_50' in df.columns
    assert 'ma_200' in df.columns

def test_normalize_data():
    """Test data normalization"""
    data = pd.Series(np.random.normal(100, 10, 100))
    normalized = normalize_data(data)
    
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert len(normalized) == len(data)

def test_calculate_returns():
    """Test returns calculation"""
    prices = pd.Series([100, 102, 99, 103, 101])
    returns = calculate_returns(prices)
    
    assert len(returns) == len(prices)
    assert pd.isna(returns[0])  # First return is NaN
    assert abs(returns[1] - 0.02) < 0.0001  # 2% return
    assert abs(returns[2] - (-0.0294)) < 0.0001  # -2.94% return

def test_calculate_volatility():
    """Test volatility calculation"""
    returns = pd.Series(np.random.normal(0, 0.02, 100))
    volatility = calculate_volatility(returns)
    
    assert len(volatility) == len(returns)
    assert all(pd.isna(volatility[:19]))  # First 19 values should be NaN
    assert all(v >= 0 for v in volatility[19:])  # All volatilities should be non-negative

def test_preprocess_market_data():
    """Test market data preprocessing"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    
    processed = preprocess_market_data(df)
    
    assert isinstance(processed.index, pd.DatetimeIndex)
    assert processed.isnull().sum().sum() == 0  # No null values
    assert len(processed) == len(df)

def test_create_features():
    """Test feature creation"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)
    
    features = create_features(df)
    
    assert 'returns' in features.columns
    assert 'volatility' in features.columns
    assert 'normalized_volume' in features.columns
    assert len(features) == len(df)

def test_split_data():
    """Test data splitting"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'close': np.random.normal(100, 10, 100)
    }, index=dates)
    
    train, test = split_data(df, train_ratio=0.8)
    
    assert len(train) == 80
    assert len(test) == 20
    assert train.index[0] < test.index[0]
    assert all(train.index < test.index[0])

def test_resample_data():
    """Test data resampling"""
    dates = pd.date_range(start='2020-01-01', periods=24, freq='H')
    df = pd.DataFrame({
        'open': np.random.normal(100, 10, 24),
        'high': np.random.normal(105, 10, 24),
        'low': np.random.normal(95, 10, 24),
        'close': np.random.normal(100, 10, 24),
        'volume': np.random.normal(1000, 100, 24)
    }, index=dates)
    
    resampled = resample_data(df, timeframe='4H')
    
    assert len(resampled) == 6  # 24 hours / 4 hours = 6 periods
    assert all(col in resampled.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_calculate_position_size():
    """Test position size calculation"""
    capital = 10000
    risk_per_trade = 0.02
    stop_loss_pct = 0.05
    
    size = calculate_position_size(capital, risk_per_trade, stop_loss_pct)
    
    assert size > 0
    assert size == (capital * risk_per_trade) / stop_loss_pct

def test_calculate_portfolio_metrics():
    """Test portfolio metrics calculation"""
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # One year of daily returns
    metrics = calculate_portfolio_metrics(returns)
    
    assert 'annual_return' in metrics
    assert 'annual_volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert metrics['annual_volatility'] > 0
    assert metrics['max_drawdown'] <= 0

def test_format_trade_data():
    """Test trade data formatting"""
    trades = [
        {
            'timestamp': '2020-01-01 10:00:00',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000,
            'amount': 0.1
        },
        {
            'timestamp': '2020-01-01 11:00:00',
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'price': 51000,
            'amount': 0.1
        }
    ]
    
    df = format_trade_data(trades)
    
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2
    assert all(col in df.columns for col in ['symbol', 'side', 'price', 'amount']) 