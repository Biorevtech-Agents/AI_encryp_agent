"""
Technical indicators module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class TechnicalIndicators:
    """
    A class for calculating technical indicators.
    """
    
    @staticmethod
    def add_moving_average(df: pd.DataFrame, window: int, column: str = 'close') -> pd.DataFrame:
        """
        Add a simple moving average to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for the moving average
            column: Column to calculate the moving average on
            
        Returns:
            DataFrame with added moving average column
        """
        df[f'ma_{window}'] = df[column].rolling(window=window).mean()
        return df

    @staticmethod
    def add_exponential_moving_average(df: pd.DataFrame, window: int, column: str = 'close') -> pd.DataFrame:
        """
        Add an exponential moving average to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for the EMA
            column: Column to calculate the EMA on
            
        Returns:
            DataFrame with added EMA column
        """
        df[f'ema_{window}'] = df[column].ewm(span=window, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14, column: str = 'close') -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for RSI calculation
            column: Column to calculate RSI on
            
        Returns:
            DataFrame with added RSI column
        """
        delta = df[column].diff()
        
        # Make two series: one for gains and one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA
        avg_gain = up.ewm(com=window-1, adjust=False).mean()
        avg_loss = down.ewm(com=window-1, adjust=False).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = 'close') -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line window
            column: Column to calculate MACD on
            
        Returns:
            DataFrame with added MACD columns
        """
        # Calculate fast and slow EMAs
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        df['macd_line'] = ema_fast - ema_slow
        
        # Calculate signal line
        df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0, column: str = 'close') -> pd.DataFrame:
        """
        Add Bollinger Bands to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for moving average
            std_dev: Number of standard deviations for the bands
            column: Column to calculate Bollinger Bands on
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        # Calculate middle band (simple moving average)
        df['bb_middle'] = df[column].rolling(window=window).mean()
        
        # Calculate standard deviation
        rolling_std = df[column].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate percent b
        df['bb_percent_b'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for ATR calculation
            
        Returns:
            DataFrame with added ATR column
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        df['atr'] = tr.ewm(span=window, adjust=False).mean()
        
        return df

    @staticmethod
    def add_stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            k_window: Window size for %K calculation
            d_window: Window size for %D calculation
            
        Returns:
            DataFrame with added Stochastic Oscillator columns
        """
        # Calculate %K
        low_min = df['low'].rolling(window=k_window).min()
        high_max = df['high'].rolling(window=k_window).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_window).mean()
        
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added OBV column
        """
        df['obv'] = np.where(df['close'] > df['close'].shift(), 
                            df['volume'], 
                            np.where(df['close'] < df['close'].shift(), 
                                    -df['volume'], 0)).cumsum()
        return df

    @staticmethod
    def add_ichimoku_cloud(df: pd.DataFrame, conversion_window: int = 9, 
                          base_window: int = 26, leading_span_b_window: int = 52, 
                          lagging_span_window: int = 26) -> pd.DataFrame:
        """
        Add Ichimoku Cloud indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            conversion_window: Window for Tenkan-sen (Conversion Line)
            base_window: Window for Kijun-sen (Base Line)
            leading_span_b_window: Window for Senkou Span B (Leading Span B)
            lagging_span_window: Window for Chikou Span (Lagging Span)
            
        Returns:
            DataFrame with added Ichimoku Cloud columns
        """
        # Calculate Tenkan-sen (Conversion Line)
        high_values = df['high'].rolling(window=conversion_window).max()
        low_values = df['low'].rolling(window=conversion_window).min()
        df['ichimoku_conversion'] = (high_values + low_values) / 2
        
        # Calculate Kijun-sen (Base Line)
        high_values = df['high'].rolling(window=base_window).max()
        low_values = df['low'].rolling(window=base_window).min()
        df['ichimoku_base'] = (high_values + low_values) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(base_window)
        
        # Calculate Senkou Span B (Leading Span B)
        high_values = df['high'].rolling(window=leading_span_b_window).max()
        low_values = df['low'].rolling(window=leading_span_b_window).min()
        df['ichimoku_span_b'] = ((high_values + low_values) / 2).shift(base_window)
        
        # Calculate Chikou Span (Lagging Span)
        df['ichimoku_lagging'] = df['close'].shift(-lagging_span_window)
        
        return df

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        df = TechnicalIndicators.add_moving_average(df, window=20)
        df = TechnicalIndicators.add_moving_average(df, window=50)
        df = TechnicalIndicators.add_moving_average(df, window=200)
        
        df = TechnicalIndicators.add_exponential_moving_average(df, window=12)
        df = TechnicalIndicators.add_exponential_moving_average(df, window=26)
        
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        df = TechnicalIndicators.add_stochastic_oscillator(df)
        df = TechnicalIndicators.add_obv(df)
        df = TechnicalIndicators.add_ichimoku_cloud(df)
        
        return df

def calculate_support_resistance(df: pd.DataFrame, window: int = 20, threshold: float = 0.01) -> Dict[str, float]:
    """
    Calculate support and resistance levels.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for identifying pivots
        threshold: Threshold for considering a level as support/resistance
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Identify pivot highs and lows
    pivot_high = df['high'].rolling(window=window, center=True).max()
    pivot_low = df['low'].rolling(window=window, center=True).min()
    
    # Find support levels (recent pivot lows)
    support_levels = pivot_low.dropna().tail(5).tolist()
    
    # Find resistance levels (recent pivot highs)
    resistance_levels = pivot_high.dropna().tail(5).tolist()
    
    # Group nearby levels
    def group_levels(levels, threshold):
        if not levels:
            return []
            
        levels = sorted(levels)
        grouped = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if level > current_group[-1] * (1 + threshold) or level < current_group[-1] * (1 - threshold):
                # Start a new group
                grouped.append(sum(current_group) / len(current_group))
                current_group = [level]
            else:
                # Add to current group
                current_group.append(level)
                
        # Add the last group
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
            
        return grouped
    
    grouped_support = group_levels(support_levels, threshold)
    grouped_resistance = group_levels(resistance_levels, threshold)
    
    return {
        'support_levels': grouped_support,
        'resistance_levels': grouped_resistance
    }

def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect common chart patterns.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary indicating detected patterns
    """
    patterns = {
        'double_top': False,
        'double_bottom': False,
        'head_and_shoulders': False,
        'inverse_head_and_shoulders': False,
        'bullish_engulfing': False,
        'bearish_engulfing': False,
        'doji': False,
        'hammer': False,
        'shooting_star': False
    }
    
    # Detect candlestick patterns (for the last candle)
    if len(df) > 1:
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Doji
        body_size = abs(curr['close'] - curr['open'])
        wick_size = curr['high'] - curr['low']
        if body_size <= 0.1 * wick_size:
            patterns['doji'] = True
        
        # Bullish Engulfing
        if (curr['open'] < prev['close'] and 
            curr['close'] > prev['open'] and 
            curr['close'] > curr['open']):
            patterns['bullish_engulfing'] = True
            
        # Bearish Engulfing
        if (curr['open'] > prev['close'] and 
            curr['close'] < prev['open'] and 
            curr['close'] < curr['open']):
            patterns['bearish_engulfing'] = True
            
        # Hammer
        body_top = max(curr['open'], curr['close'])
        body_bottom = min(curr['open'], curr['close'])
        if (body_size > 0 and
            (curr['high'] - body_top) < body_size * 0.5 and
            (body_bottom - curr['low']) > body_size * 2 and
            curr['close'] > curr['open']):
            patterns['hammer'] = True
            
        # Shooting Star
        if (body_size > 0 and
            (curr['high'] - body_top) > body_size * 2 and
            (body_bottom - curr['low']) < body_size * 0.5 and
            curr['close'] < curr['open']):
            patterns['shooting_star'] = True
    
    # More complex patterns requiring more historical data
    if len(df) > 30:
        # Simple double top detection
        highs = df['high'].rolling(window=5, center=True).max()
        if (highs.iloc[-15:-5].max() > highs.iloc[-5:].max() * 0.98 and
            highs.iloc[-15:-5].max() < highs.iloc[-5:].max() * 1.02 and
            highs.iloc[-15:-5].idxmax() != highs.iloc[-5:].idxmax()):
            patterns['double_top'] = True
            
        # Simple double bottom detection
        lows = df['low'].rolling(window=5, center=True).min()
        if (lows.iloc[-15:-5].min() > lows.iloc[-5:].min() * 0.98 and
            lows.iloc[-15:-5].min() < lows.iloc[-5:].min() * 1.02 and
            lows.iloc[-15:-5].idxmin() != lows.iloc[-5:].idxmin()):
            patterns['double_bottom'] = True
    
    return patterns 