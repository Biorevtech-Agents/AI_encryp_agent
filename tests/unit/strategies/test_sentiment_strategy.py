"""
Tests for the sentiment-based trading strategy.
"""

import pytest
from src.strategies.sentiment_strategy import SentimentStrategy

def test_sentiment_strategy_initialization():
    """Test sentiment strategy initialization"""
    strategy = SentimentStrategy(sentiment_weight=0.4)
    assert strategy.sentiment_weight == 0.4
    assert isinstance(strategy.thresholds, dict)
    assert "RSI_OVERBOUGHT" in strategy.thresholds
    assert "RSI_OVERSOLD" in strategy.thresholds

def test_generate_signals(sample_ohlcv_data, sample_sentiment_data):
    """Test signal generation with both market and sentiment data"""
    strategy = SentimentStrategy()
    signals = strategy.generate_signals(sample_ohlcv_data, sample_sentiment_data)
    
    # Check technical signals
    assert "RSI" in signals
    assert "MACD" in signals
    assert "MA_50" in signals
    assert "MA_200" in signals
    assert "BB_POSITION" in signals
    
    # Check sentiment signals
    assert "SENTIMENT_SCORE" in signals
    assert signals["SENTIMENT_SCORE"] == sample_sentiment_data["score"]
    assert signals["SENTIMENT_MAGNITUDE"] == sample_sentiment_data["magnitude"]
    assert signals["NEWS_SENTIMENT"] == sample_sentiment_data["news_sentiment"]
    assert signals["SOCIAL_SENTIMENT"] == sample_sentiment_data["social_sentiment"]

def test_generate_signals_no_sentiment(sample_ohlcv_data):
    """Test signal generation with only market data"""
    strategy = SentimentStrategy()
    signals = strategy.generate_signals(sample_ohlcv_data)
    
    # Check technical signals exist
    assert "RSI" in signals
    assert "MACD" in signals
    assert "MA_50" in signals
    assert "MA_200" in signals
    
    # Check sentiment signals don't exist
    assert "SENTIMENT_SCORE" not in signals
    assert "NEWS_SENTIMENT" not in signals
    assert "SOCIAL_SENTIMENT" not in signals

def test_make_decision_buy_signal(sample_ohlcv_data, sample_sentiment_data):
    """Test decision making with bullish signals"""
    strategy = SentimentStrategy(sentiment_weight=0.5)
    signals = strategy.generate_signals(sample_ohlcv_data, sample_sentiment_data)
    
    # Override some signals to force a buy decision
    signals["RSI"] = 35  # Oversold
    signals["MACD_CROSSOVER"] = "bullish"
    signals["SENTIMENT_SCORE"] = 0.8  # Very bullish
    
    decision = strategy.make_decision(signals, risk_tolerance=0.7)
    
    assert decision["action"] == "buy"
    assert decision["confidence"] > 0.5
    assert "technical_score" in decision
    assert "sentiment_score" in decision
    assert "combined_score" in decision

def test_make_decision_sell_signal(sample_ohlcv_data, sample_sentiment_data):
    """Test decision making with bearish signals"""
    strategy = SentimentStrategy(sentiment_weight=0.5)
    signals = strategy.generate_signals(sample_ohlcv_data, sample_sentiment_data)
    
    # Override some signals to force a sell decision
    signals["RSI"] = 75  # Overbought
    signals["MACD_CROSSOVER"] = "bearish"
    signals["SENTIMENT_SCORE"] = -0.8  # Very bearish
    
    decision = strategy.make_decision(signals, risk_tolerance=0.7)
    
    assert decision["action"] == "sell"
    assert decision["confidence"] > 0.5
    assert "technical_score" in decision
    assert "sentiment_score" in decision
    assert "combined_score" in decision

def test_make_decision_hold_signal(sample_ohlcv_data, sample_sentiment_data):
    """Test decision making with neutral signals"""
    strategy = SentimentStrategy(sentiment_weight=0.5)
    signals = strategy.generate_signals(sample_ohlcv_data, sample_sentiment_data)
    
    # Override some signals to force a hold decision
    signals["RSI"] = 50  # Neutral
    signals["MACD_CROSSOVER"] = "neutral"
    signals["SENTIMENT_SCORE"] = 0.1  # Slightly bullish
    
    decision = strategy.make_decision(signals, risk_tolerance=0.3)
    
    assert decision["action"] == "hold"
    assert "technical_score" in decision
    assert "sentiment_score" in decision
    assert "combined_score" in decision

def test_adapt_from_reflection():
    """Test strategy adaptation based on reflection"""
    strategy = SentimentStrategy(sentiment_weight=0.5)
    initial_weight = strategy.sentiment_weight
    initial_rsi_oversold = strategy.thresholds["RSI_OVERSOLD"]
    
    reflection = {
        "win_rate": 0.35,  # Poor performance
        "insights": [
            "RSI oversold signals were successful",
            "MACD signals often failed"
        ]
    }
    
    strategy.adapt_from_reflection(reflection)
    
    # Check that parameters were adjusted
    assert strategy.sentiment_weight != initial_weight
    assert strategy.thresholds["RSI_OVERSOLD"] < initial_rsi_oversold
    assert strategy.thresholds["MACD_SIGNAL"] > 0

def test_calculate_technical_score():
    """Test technical score calculation"""
    strategy = SentimentStrategy()
    
    signals = {
        "RSI": 25,  # Oversold - bullish
        "MACD_CROSSOVER": "bullish",
        "MA_CROSSOVER": "bullish",
        "BB_POSITION": "lower",
        "VOLUME_RATIO": 2.0
    }
    
    score = strategy._calculate_technical_score(signals)
    assert -1 <= score <= 1
    assert score > 0  # Should be bullish

def test_calculate_sentiment_score():
    """Test sentiment score calculation"""
    strategy = SentimentStrategy()
    
    signals = {
        "SENTIMENT_SCORE": 0.8,
        "NEWS_SENTIMENT": 0.7,
        "SOCIAL_SENTIMENT": 0.9,
        "SENTIMENT_MAGNITUDE": 0.9
    }
    
    score = strategy._calculate_sentiment_score(signals)
    assert -1 <= score <= 1
    assert score > 0  # Should be bullish 