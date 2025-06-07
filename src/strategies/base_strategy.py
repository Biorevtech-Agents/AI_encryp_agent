import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('base_strategy')

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        self.current_positions = {}  # Symbol -> Position info
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Generate trading signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to their OHLCV DataFrames
            
        Returns:
            Dictionary mapping symbols to signals ('buy', 'sell', 'hold')
        """
        pass
    
    def calculate_position_size(self, symbol: str, signal: str, 
                               portfolio_value: float, risk_per_trade: float,
                               current_price: float, stop_loss_price: Optional[float] = None) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            symbol: Trading pair symbol
            signal: Trading signal ('buy' or 'sell')
            portfolio_value: Current portfolio value
            risk_per_trade: Maximum risk per trade as percentage of portfolio
            current_price: Current price of the asset
            stop_loss_price: Price at which to place stop loss (optional)
            
        Returns:
            Position size in base currency units
        """
        if signal not in ['buy', 'sell']:
            return 0.0
            
        # Default risk calculation (fixed percentage of portfolio)
        position_value = portfolio_value * risk_per_trade
        
        # If stop loss is provided, adjust position size based on risk
        if stop_loss_price is not None and stop_loss_price > 0:
            # Calculate risk per unit
            if signal == 'buy':
                risk_per_unit = current_price - stop_loss_price
            else:  # signal == 'sell'
                risk_per_unit = stop_loss_price - current_price
                
            # Adjust position size if risk per unit is valid
            if risk_per_unit > 0:
                max_units = position_value / risk_per_unit
                return max_units
        
        # Default to simple percentage of portfolio
        return position_value / current_price
    
    def calculate_stop_loss(self, symbol: str, signal: str, 
                           current_price: float, atr: Optional[float] = None,
                           support_resistance: Optional[Dict] = None,
                           stop_loss_percentage: float = 0.05) -> float:
        """
        Calculate stop loss price for a trade.
        
        Args:
            symbol: Trading pair symbol
            signal: Trading signal ('buy' or 'sell')
            current_price: Current price of the asset
            atr: Average True Range value (optional)
            support_resistance: Support and resistance levels (optional)
            stop_loss_percentage: Default stop loss percentage
            
        Returns:
            Stop loss price
        """
        # Default stop loss based on percentage
        if signal == 'buy':
            stop_loss = current_price * (1 - stop_loss_percentage)
        else:  # signal == 'sell'
            stop_loss = current_price * (1 + stop_loss_percentage)
            
        # If ATR is available, use it for dynamic stop loss
        if atr is not None:
            atr_multiplier = 2.0  # Typically 2-3 times ATR
            if signal == 'buy':
                atr_stop = current_price - (atr * atr_multiplier)
                stop_loss = max(stop_loss, atr_stop)  # Use the higher of the two
            else:  # signal == 'sell'
                atr_stop = current_price + (atr * atr_multiplier)
                stop_loss = min(stop_loss, atr_stop)  # Use the lower of the two
                
        # If support/resistance levels are available, use them
        if support_resistance is not None:
            if signal == 'buy' and 'support_levels' in support_resistance:
                # Find the nearest support level below current price
                supports_below = [s for s in support_resistance['support_levels'] if s < current_price]
                if supports_below:
                    nearest_support = max(supports_below)
                    stop_loss = max(stop_loss, nearest_support * 0.99)  # Just below support
                    
            elif signal == 'sell' and 'resistance_levels' in support_resistance:
                # Find the nearest resistance level above current price
                resistances_above = [r for r in support_resistance['resistance_levels'] if r > current_price]
                if resistances_above:
                    nearest_resistance = min(resistances_above)
                    stop_loss = min(stop_loss, nearest_resistance * 1.01)  # Just above resistance
        
        return stop_loss
    
    def calculate_take_profit(self, symbol: str, signal: str, 
                             current_price: float, stop_loss_price: float,
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price for a trade.
        
        Args:
            symbol: Trading pair symbol
            signal: Trading signal ('buy' or 'sell')
            current_price: Current price of the asset
            stop_loss_price: Stop loss price
            risk_reward_ratio: Risk to reward ratio (e.g., 1:2 means take profit is twice the distance of stop loss)
            
        Returns:
            Take profit price
        """
        # Calculate risk (distance to stop loss)
        if signal == 'buy':
            risk = current_price - stop_loss_price
            take_profit = current_price + (risk * risk_reward_ratio)
        else:  # signal == 'sell'
            risk = stop_loss_price - current_price
            take_profit = current_price - (risk * risk_reward_ratio)
            
        return take_profit
    
    def evaluate_performance(self, trades: List[Dict]) -> Dict:
        """
        Evaluate the performance of the strategy.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('profit', 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_profit = sum(t.get('profit', 0) for t in trades) / total_trades if total_trades > 0 else 0
        
        # Calculate drawdown
        equity_curve = []
        current_equity = 1000  # Starting equity
        
        for trade in trades:
            current_equity += trade.get('profit', 0)
            equity_curve.append(current_equity)
            
        if equity_curve:
            max_equity = 1000  # Starting equity
            max_drawdown = 0
            
            for equity in equity_curve:
                max_equity = max(max_equity, equity)
                drawdown = (max_equity - equity) / max_equity
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
            
        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t.get('profit', 0) for t in trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_profit': average_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def update_position(self, symbol: str, action: str, price: float, 
                       size: float, timestamp: pd.Timestamp) -> Dict:
        """
        Update the current position for a symbol.
        
        Args:
            symbol: Trading pair symbol
            action: Trade action ('buy', 'sell', 'close')
            price: Execution price
            size: Position size
            timestamp: Execution timestamp
            
        Returns:
            Updated position information
        """
        position = self.current_positions.get(symbol, {
            'symbol': symbol,
            'position_type': None,
            'entry_price': 0,
            'current_price': 0,
            'size': 0,
            'unrealized_pnl': 0,
            'entry_time': None,
            'last_update_time': None
        })
        
        if action == 'buy':
            # Opening or adding to long position
            if position['position_type'] in [None, 'long']:
                # Calculate new average entry price if adding to position
                if position['size'] > 0:
                    total_value = (position['entry_price'] * position['size']) + (price * size)
                    total_size = position['size'] + size
                    avg_price = total_value / total_size
                else:
                    avg_price = price
                    total_size = size
                    
                position.update({
                    'position_type': 'long',
                    'entry_price': avg_price,
                    'current_price': price,
                    'size': total_size,
                    'entry_time': position['entry_time'] or timestamp,
                    'last_update_time': timestamp
                })
            else:
                # Reducing short position
                remaining_size = position['size'] - size
                
                if remaining_size <= 0:
                    # Position closed or flipped
                    if remaining_size < 0:
                        # Flipped to long
                        position.update({
                            'position_type': 'long',
                            'entry_price': price,
                            'current_price': price,
                            'size': abs(remaining_size),
                            'entry_time': timestamp,
                            'last_update_time': timestamp
                        })
                    else:
                        # Position closed
                        position.update({
                            'position_type': None,
                            'entry_price': 0,
                            'current_price': price,
                            'size': 0,
                            'entry_time': None,
                            'last_update_time': timestamp
                        })
                else:
                    # Still short, just smaller position
                    position.update({
                        'current_price': price,
                        'size': remaining_size,
                        'last_update_time': timestamp
                    })
                    
        elif action == 'sell':
            # Opening or adding to short position
            if position['position_type'] in [None, 'short']:
                # Calculate new average entry price if adding to position
                if position['size'] > 0:
                    total_value = (position['entry_price'] * position['size']) + (price * size)
                    total_size = position['size'] + size
                    avg_price = total_value / total_size
                else:
                    avg_price = price
                    total_size = size
                    
                position.update({
                    'position_type': 'short',
                    'entry_price': avg_price,
                    'current_price': price,
                    'size': total_size,
                    'entry_time': position['entry_time'] or timestamp,
                    'last_update_time': timestamp
                })
            else:
                # Reducing long position
                remaining_size = position['size'] - size
                
                if remaining_size <= 0:
                    # Position closed or flipped
                    if remaining_size < 0:
                        # Flipped to short
                        position.update({
                            'position_type': 'short',
                            'entry_price': price,
                            'current_price': price,
                            'size': abs(remaining_size),
                            'entry_time': timestamp,
                            'last_update_time': timestamp
                        })
                    else:
                        # Position closed
                        position.update({
                            'position_type': None,
                            'entry_price': 0,
                            'current_price': price,
                            'size': 0,
                            'entry_time': None,
                            'last_update_time': timestamp
                        })
                else:
                    # Still long, just smaller position
                    position.update({
                        'current_price': price,
                        'size': remaining_size,
                        'last_update_time': timestamp
                    })
                    
        elif action == 'close':
            # Close entire position
            position.update({
                'position_type': None,
                'entry_price': 0,
                'current_price': price,
                'size': 0,
                'unrealized_pnl': 0,
                'entry_time': None,
                'last_update_time': timestamp
            })
            
        # Update unrealized PnL
        if position['position_type'] == 'long' and position['size'] > 0:
            position['unrealized_pnl'] = (position['current_price'] - position['entry_price']) * position['size']
        elif position['position_type'] == 'short' and position['size'] > 0:
            position['unrealized_pnl'] = (position['entry_price'] - position['current_price']) * position['size']
        else:
            position['unrealized_pnl'] = 0
            
        # Update the position in the dictionary
        self.current_positions[symbol] = position
        
        return position
    
    def update_prices(self, prices: Dict[str, float], timestamp: pd.Timestamp) -> None:
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary mapping symbols to their current prices
            timestamp: Current timestamp
        """
        for symbol, price in prices.items():
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                position['current_price'] = price
                position['last_update_time'] = timestamp
                
                # Update unrealized PnL
                if position['position_type'] == 'long' and position['size'] > 0:
                    position['unrealized_pnl'] = (position['current_price'] - position['entry_price']) * position['size']
                elif position['position_type'] == 'short' and position['size'] > 0:
                    position['unrealized_pnl'] = (position['entry_price'] - position['current_price']) * position['size']
                    
                self.current_positions[symbol] = position 