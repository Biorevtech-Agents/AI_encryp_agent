"""
Pytest configuration file with common fixtures.
"""

import pytest
from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_data() -> List[Dict[str, float]]:
    """Generate sample OHLCV data for testing"""
    data = []
    base_price = 50000.0
    timestamp = datetime.now()
    
    for i in range(100):
        # Generate random price movements
        price_change = np.random.normal(0, 100)
        high = base_price + abs(price_change)
        low = base_price - abs(price_change)
        close = base_price + price_change
        volume = np.random.uniform(1, 10) * 100
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'open': base_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        base_price = close
        timestamp += timedelta(hours=1)
    
    return data

@pytest.fixture
def sample_sentiment_data() -> Dict[str, Any]:
    """Generate sample sentiment data for testing"""
    return {
        'score': 0.75,
        'magnitude': 0.8,
        'trend': 'bullish',
        'news_sentiment': 0.6,
        'social_sentiment': 0.8,
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def mock_market_data(sample_ohlcv_data):
    """Create a mock market data fetcher"""
    class MockMarketDataFetcher:
        def __init__(self):
            self.data = sample_ohlcv_data
        
        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, float]]:
            return self.data[-limit:]
        
        def get_current_price(self, symbol: str) -> float:
            return self.data[-1]['close']
        
        def get_ohlcv_data(self, symbol: str, timeframe: str) -> List[Dict[str, float]]:
            return self.data
    
    return MockMarketDataFetcher()

@pytest.fixture
def mock_sentiment_analyzer(sample_sentiment_data):
    """Create a mock sentiment analyzer"""
    class MockSentimentAnalyzer:
        def __init__(self):
            self.data = sample_sentiment_data
        
        def get_sentiment(self, symbol: str) -> Dict[str, Any]:
            return self.data
        
        def update_sentiment(self, symbol: str) -> None:
            pass
    
    return MockSentimentAnalyzer() 