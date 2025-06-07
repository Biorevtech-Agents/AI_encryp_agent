"""
Tests for the trading agent functionality.
"""

import pytest
from datetime import datetime
from src.agent.trading_agent import TradingAgent
from src.strategies.sentiment_strategy import SentimentStrategy

@pytest.fixture
def trading_agent():
    """Create a trading agent instance for testing"""
    strategy = SentimentStrategy(sentiment_weight=0.5)
    return TradingAgent(
        strategy=strategy,
        initial_balance=10000,
        risk_tolerance=0.5,
        memory_decay_rate=0.9
    )

def test_trading_agent_initialization(trading_agent):
    """Test trading agent initialization"""
    assert isinstance(trading_agent.strategy, SentimentStrategy)
    assert trading_agent.balance == 10000
    assert trading_agent.risk_tolerance == 0.5
    assert trading_agent.memory_decay_rate == 0.9
    assert isinstance(trading_agent.portfolio, dict)

def test_analyze_market(trading_agent, mock_market_data, mock_sentiment_analyzer):
    """Test market analysis functionality"""
    analysis = trading_agent.analyze_market(
        symbol='BTC/USDT',
        market_data=mock_market_data,
        sentiment_analyzer=mock_sentiment_analyzer
    )
    
    assert isinstance(analysis, dict)
    assert 'signals' in analysis
    assert 'technical_indicators' in analysis['signals']
    assert 'sentiment_indicators' in analysis['signals']
    assert 'market_state' in analysis

def test_execute_trade(trading_agent):
    """Test trade execution"""
    trade = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'price': 50000,
        'amount': 0.1,
        'confidence': 0.8,
        'stop_loss': 48000,
        'take_profit': 53000
    }
    
    result = trading_agent.execute_trade(trade)
    
    assert isinstance(result, dict)
    assert 'success' in result
    assert 'message' in result
    assert 'trade_id' in result
    
    # Check portfolio update
    assert 'BTC/USDT' in trading_agent.portfolio
    assert trading_agent.portfolio['BTC/USDT']['amount'] == 0.1
    assert trading_agent.balance == 5000  # 10000 - (50000 * 0.1)

def test_monitor_positions(trading_agent, mock_market_data):
    """Test position monitoring"""
    # Add a test position
    trade = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'price': 50000,
        'amount': 0.1,
        'confidence': 0.8,
        'stop_loss': 48000,
        'take_profit': 53000
    }
    trading_agent.execute_trade(trade)
    
    # Test monitoring with price above take profit
    mock_market_data.current_price = 54000
    actions = trading_agent.monitor_positions(mock_market_data)
    
    assert isinstance(actions, list)
    assert len(actions) > 0
    assert actions[0]['action'] == 'sell'
    assert actions[0]['reason'] == 'take_profit'

def test_adapt_strategy(trading_agent):
    """Test strategy adaptation"""
    # Add some test trades to memory
    trades = [
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'price': 50000,
            'amount': 0.1,
            'confidence': 0.8,
            'result': 'success',
            'profit_loss': 500,
            'signals': {
                'RSI': 30,
                'SENTIMENT_SCORE': 0.7
            }
        },
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTC/USDT',
            'action': 'sell',
            'price': 48000,
            'amount': 0.1,
            'confidence': 0.6,
            'result': 'failure',
            'profit_loss': -200,
            'signals': {
                'RSI': 70,
                'SENTIMENT_SCORE': -0.3
            }
        }
    ]
    
    for trade in trades:
        trading_agent.update_memory(trade)
    
    # Test strategy adaptation
    initial_weight = trading_agent.strategy.sentiment_weight
    trading_agent.adapt_strategy()
    
    assert trading_agent.strategy.sentiment_weight != initial_weight

def test_calculate_performance_metrics(trading_agent):
    """Test performance metrics calculation"""
    # Add some test trades
    trades = [
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'price': 50000,
            'amount': 0.1,
            'profit_loss': 500
        },
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTC/USDT',
            'action': 'sell',
            'price': 48000,
            'amount': 0.1,
            'profit_loss': -200
        }
    ]
    
    for trade in trades:
        trading_agent.update_memory(trade)
    
    metrics = trading_agent.calculate_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert 'total_profit_loss' in metrics
    assert 'win_rate' in metrics
    assert 'average_profit' in metrics
    assert 'average_loss' in metrics
    assert 'risk_reward_ratio' in metrics

def test_handle_market_event(trading_agent, mock_market_data, mock_sentiment_analyzer):
    """Test market event handling"""
    event = {
        'type': 'price_change',
        'symbol': 'BTC/USDT',
        'data': {
            'price': 52000,
            'volume': 100,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    response = trading_agent.handle_market_event(
        event,
        market_data=mock_market_data,
        sentiment_analyzer=mock_sentiment_analyzer
    )
    
    assert isinstance(response, dict)
    assert 'action_taken' in response
    assert 'analysis' in response
    assert 'position_updates' in response

def test_validate_and_adjust_trade(trading_agent):
    """Test trade validation and adjustment"""
    trade = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'price': 50000,
        'amount': 0.3,  # Too large for current balance
        'confidence': 0.8
    }
    
    adjusted_trade = trading_agent.validate_and_adjust_trade(trade)
    
    assert adjusted_trade['amount'] < trade['amount']
    assert adjusted_trade['price'] * adjusted_trade['amount'] <= trading_agent.balance
    assert 'stop_loss' in adjusted_trade
    assert 'take_profit' in adjusted_trade

def test_emergency_stop(trading_agent, mock_market_data):
    """Test emergency stop functionality"""
    # Add some positions
    trade = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'price': 50000,
        'amount': 0.1,
        'confidence': 0.8
    }
    trading_agent.execute_trade(trade)
    
    # Test emergency stop
    result = trading_agent.emergency_stop(mock_market_data)
    
    assert isinstance(result, dict)
    assert 'success' in result
    assert 'positions_closed' in result
    assert len(trading_agent.portfolio) == 0  # All positions should be closed 