import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple

from src.utils.technical_indicators import TechnicalIndicators, calculate_support_resistance, detect_patterns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('technical_strategy')

class TechnicalStrategy:
    """Trading strategy based on technical analysis."""
    
    def __init__(self, name: str = "Technical Strategy"):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        self.current_positions = {}  # Symbol -> Position info
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Generate trading signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to their OHLCV DataFrames
            
        Returns:
            Dictionary mapping symbols to signals ('buy', 'sell', 'hold')
        """
        signals = {}
        
        for symbol, df in data.items():
            if df.empty or len(df) < 50:
                signals[symbol] = 'hold'
                continue
                
            # Add technical indicators
            df_with_indicators = TechnicalIndicators.add_all_indicators(df)
            
            # Calculate support and resistance levels
            support_resistance = calculate_support_resistance(df)
            
            # Detect chart patterns
            patterns = detect_patterns(df)
            
            # Generate signal based on multiple indicators
            signal = self._generate_signal_for_symbol(df_with_indicators, support_resistance, patterns)
            signals[symbol] = signal
            
            logger.info(f"Generated signal for {symbol}: {signal}")
            
        return signals
    
    def _generate_signal_for_symbol(self, df: pd.DataFrame, 
                                   support_resistance: Dict, 
                                   patterns: Dict) -> str:
        """
        Generate a trading signal for a single symbol.
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            support_resistance: Dictionary with support and resistance levels
            patterns: Dictionary with detected chart patterns
            
        Returns:
            Trading signal ('buy', 'sell', 'hold')
        """
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Initialize signal scores
        buy_score = 0
        sell_score = 0
        
        # 1. Trend analysis using moving averages
        if 'ma_50' in df.columns and 'ma_200' in df.columns:
            # Golden cross (short-term MA crosses above long-term MA)
            if df['ma_50'].iloc[-2] <= df['ma_200'].iloc[-2] and latest['ma_50'] > latest['ma_200']:
                buy_score += 2
            # Death cross (short-term MA crosses below long-term MA)
            elif df['ma_50'].iloc[-2] >= df['ma_200'].iloc[-2] and latest['ma_50'] < latest['ma_200']:
                sell_score += 2
            # Price above both MAs and MAs are aligned (uptrend)
            elif latest['close'] > latest['ma_50'] > latest['ma_200']:
                buy_score += 1
            # Price below both MAs and MAs are aligned (downtrend)
            elif latest['close'] < latest['ma_50'] < latest['ma_200']:
                sell_score += 1
        
        # 2. Momentum analysis using RSI
        if 'rsi' in df.columns:
            # Oversold condition
            if latest['rsi'] < 30:
                buy_score += 1
            # Overbought condition
            elif latest['rsi'] > 70:
                sell_score += 1
            # RSI crossing above 50 (bullish momentum)
            elif df['rsi'].iloc[-2] < 50 and latest['rsi'] > 50:
                buy_score += 1
            # RSI crossing below 50 (bearish momentum)
            elif df['rsi'].iloc[-2] > 50 and latest['rsi'] < 50:
                sell_score += 1
        
        # 3. MACD analysis
        if 'macd_line' in df.columns and 'macd_signal' in df.columns:
            # MACD line crosses above signal line (bullish)
            if df['macd_line'].iloc[-2] <= df['macd_signal'].iloc[-2] and latest['macd_line'] > latest['macd_signal']:
                buy_score += 1
            # MACD line crosses below signal line (bearish)
            elif df['macd_line'].iloc[-2] >= df['macd_signal'].iloc[-2] and latest['macd_line'] < latest['macd_signal']:
                sell_score += 1
            # MACD histogram increasing (momentum building)
            if latest['macd_histogram'] > df['macd_histogram'].iloc[-2] > 0:
                buy_score += 0.5
            # MACD histogram decreasing (momentum weakening)
            elif latest['macd_histogram'] < df['macd_histogram'].iloc[-2] < 0:
                sell_score += 0.5
        
        # 4. Bollinger Bands analysis
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            # Price touching lower band (potential bounce)
            if latest['close'] <= latest['bb_lower'] * 1.01:
                buy_score += 1
            # Price touching upper band (potential reversal)
            elif latest['close'] >= latest['bb_upper'] * 0.99:
                sell_score += 1
            # Price breaking out above upper band (strong momentum)
            elif latest['close'] > latest['bb_upper'] and df['close'].iloc[-2] <= df['bb_upper'].iloc[-2]:
                buy_score += 0.5  # Could also be a sell signal depending on other factors
            # Price breaking below lower band (strong downward momentum)
            elif latest['close'] < latest['bb_lower'] and df['close'].iloc[-2] >= df['bb_lower'].iloc[-2]:
                sell_score += 0.5  # Could also be a buy signal depending on other factors
        
        # 5. Support and resistance analysis
        if support_resistance:
            current_price = latest['close']
            
            # Check if price is near support level
            for support in support_resistance.get('support_levels', []):
                if 0.98 * support <= current_price <= 1.02 * support:
                    buy_score += 1
                    break
                    
            # Check if price is near resistance level
            for resistance in support_resistance.get('resistance_levels', []):
                if 0.98 * resistance <= current_price <= 1.02 * resistance:
                    sell_score += 1
                    break
        
        # 6. Chart pattern analysis
        if patterns:
            # Bullish patterns
            if patterns.get('double_bottom', False) or patterns.get('inverse_head_and_shoulders', False):
                buy_score += 1
            if patterns.get('bullish_engulfing', False) or patterns.get('hammer', False):
                buy_score += 0.5
                
            # Bearish patterns
            if patterns.get('double_top', False) or patterns.get('head_and_shoulders', False):
                sell_score += 1
            if patterns.get('bearish_engulfing', False) or patterns.get('shooting_star', False):
                sell_score += 0.5
                
            # Neutral patterns
            if patterns.get('doji', False):
                # In an uptrend, doji can be bearish
                if latest['close'] > latest.get('ma_50', 0):
                    sell_score += 0.3
                # In a downtrend, doji can be bullish
                else:
                    buy_score += 0.3
        
        # 7. Volume analysis
        if 'volume' in df.columns and 'obv' in df.columns:
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            # High volume breakout
            if latest['volume'] > 1.5 * avg_volume:
                if latest['close'] > latest['open']:
                    buy_score += 0.5
                else:
                    sell_score += 0.5
                    
            # OBV trend
            if df['obv'].iloc[-5:].is_monotonic_increasing:
                buy_score += 0.5
            elif df['obv'].iloc[-5:].is_monotonic_decreasing:
                sell_score += 0.5
        
        # 8. Ichimoku Cloud analysis
        if ('ichimoku_conversion_line' in df.columns and 
            'ichimoku_base_line' in df.columns and 
            'ichimoku_leading_span_a' in df.columns and 
            'ichimoku_leading_span_b' in df.columns):
            
            # Price above the cloud
            if (latest['close'] > latest['ichimoku_leading_span_a'] and 
                latest['close'] > latest['ichimoku_leading_span_b']):
                buy_score += 1
            # Price below the cloud
            elif (latest['close'] < latest['ichimoku_leading_span_a'] and 
                  latest['close'] < latest['ichimoku_leading_span_b']):
                sell_score += 1
                
            # Conversion line crosses above base line
            if (df['ichimoku_conversion_line'].iloc[-2] <= df['ichimoku_base_line'].iloc[-2] and 
                latest['ichimoku_conversion_line'] > latest['ichimoku_base_line']):
                buy_score += 1
            # Conversion line crosses below base line
            elif (df['ichimoku_conversion_line'].iloc[-2] >= df['ichimoku_base_line'].iloc[-2] and 
                  latest['ichimoku_conversion_line'] < latest['ichimoku_base_line']):
                sell_score += 1
        
        # Determine final signal based on scores
        if buy_score > sell_score and buy_score >= 3:
            return 'buy'
        elif sell_score > buy_score and sell_score >= 3:
            return 'sell'
        else:
            return 'hold'
    
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