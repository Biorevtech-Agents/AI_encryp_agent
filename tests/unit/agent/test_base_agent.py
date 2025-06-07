"""
Tests for the base agent functionality.
"""

import pytest
from datetime import datetime
from src.agent.base_agent import BaseAgent

@pytest.fixture
def base_agent():
    """Fixture to create a base agent for testing"""
    return BaseAgent(
        initial_balance=10000,
        risk_tolerance=0.5,
        memory_decay_rate=0.9
    )

def test_base_agent_initialization(base_agent):
    """Test base agent initialization"""
    assert base_agent.balance == 10000
    assert base_agent.risk_tolerance == 0.5
    assert base_agent.memory_decay_rate == 0.9
    assert isinstance(base_agent.memory, list)
    assert isinstance(base_agent.portfolio, dict)

def test_update_memory():
    """Test memory update and decay"""
    agent = BaseAgent(memory_decay_rate=0.9)
    
    # Add a trade memory
    trade_memory = {
        'timestamp': datetime.now().isoformat(),
        'action': 'buy',
        'symbol': 'BTC/USDT',
        'price': 50000,
        'amount': 0.1,
        'confidence': 0.8,
        'signals': {
            'RSI': 30,
            'SENTIMENT_SCORE': 0.7
        }
    }
    
    agent.update_memory(trade_memory)
    assert len(agent.memory) == 1
    assert agent.memory[0]['importance'] == 1.0
    
    # Add another memory and check decay
    agent.update_memory(trade_memory)
    assert len(agent.memory) == 2
    assert agent.memory[0]['importance'] == 1.0
    assert agent.memory[1]['importance'] == 0.9

def test_reflect_on_performance():
    """Test agent's reflection capabilities"""
    agent = BaseAgent()
    
    # Add some test memories
    memories = [
        {
            'timestamp': datetime.now().isoformat(),
            'action': 'buy',
            'symbol': 'BTC/USDT',
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
            'action': 'sell',
            'symbol': 'BTC/USDT',
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
    
    for memory in memories:
        agent.update_memory(memory)
    
    reflection = agent.reflect_on_performance()
    
    assert isinstance(reflection, dict)
    assert 'win_rate' in reflection
    assert 'total_profit_loss' in reflection
    assert 'insights' in reflection
    assert isinstance(reflection['insights'], list)

def test_update_portfolio():
    """Test portfolio management"""
    agent = BaseAgent(initial_balance=10000)
    
    # Test buy
    agent.update_portfolio('BTC/USDT', 'buy', 50000, 0.1)
    assert 'BTC/USDT' in agent.portfolio
    assert agent.portfolio['BTC/USDT']['amount'] == 0.1
    assert agent.balance == 5000  # 10000 - (50000 * 0.1)
    
    # Test sell
    agent.update_portfolio('BTC/USDT', 'sell', 55000, 0.05)
    assert agent.portfolio['BTC/USDT']['amount'] == 0.05
    assert agent.balance == 7750  # 5000 + (55000 * 0.05)

def test_calculate_position_size():
    """Test position size calculation"""
    agent = BaseAgent(initial_balance=10000)
    
    # Test with different confidence levels
    size_high_conf = agent.calculate_position_size('BTC/USDT', 50000, 0.9)
    size_low_conf = agent.calculate_position_size('BTC/USDT', 50000, 0.3)
    
    assert size_high_conf > size_low_conf
    assert size_high_conf <= agent.balance
    assert size_low_conf <= agent.balance

def test_risk_management():
    """Test risk management functionality"""
    agent = BaseAgent(initial_balance=10000)
    
    # Test stop loss calculation
    stop_loss = agent.calculate_stop_loss('BTC/USDT', 50000, 'buy')
    assert stop_loss < 50000
    
    # Test take profit calculation
    take_profit = agent.calculate_take_profit('BTC/USDT', 50000, 'buy')
    assert take_profit > 50000
    
    # Test risk per trade calculation
    risk = agent.calculate_risk_per_trade('BTC/USDT', 50000, 0.8)
    assert risk <= agent.balance * agent.risk_tolerance

def test_validate_trade():
    """Test trade validation"""
    agent = BaseAgent(initial_balance=10000)
    
    # Test valid trade
    valid_trade = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'price': 50000,
        'amount': 0.1,
        'confidence': 0.8
    }
    assert agent.validate_trade(valid_trade)
    
    # Test invalid trade (insufficient balance)
    invalid_trade = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'price': 50000,
        'amount': 1.0,  # Would cost 50000, more than balance
        'confidence': 0.8
    }
    assert not agent.validate_trade(invalid_trade)

def test_memory_pruning():
    """Test memory pruning functionality"""
    agent = BaseAgent(memory_decay_rate=0.5)
    
    # Add multiple memories
    for i in range(10):
        memory = {
            'timestamp': datetime.now().isoformat(),
            'action': 'buy',
            'symbol': 'BTC/USDT',
            'price': 50000,
            'amount': 0.1,
            'confidence': 0.8
        }
        agent.update_memory(memory)
    
    # Force memory pruning
    agent.prune_memory(importance_threshold=0.3)
    
    # Check that low importance memories were removed
    assert len(agent.memory) < 10
    for memory in agent.memory:
        assert memory['importance'] >= 0.3 