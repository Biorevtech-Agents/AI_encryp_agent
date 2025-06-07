"""
Sentiment-based trading strategy that combines technical indicators with sentiment analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.strategies.base_strategy import BaseStrategy
from src.utils.technical_indicators import TechnicalIndicators

class SentimentStrategy(BaseStrategy):
    """
    Trading strategy that combines technical indicators with sentiment analysis
    to make trading decisions.
    """
    
    def __init__(self, sentiment_weight: float = 0.3):
        """
        Initialize the sentiment strategy.
        
        Args:
            sentiment_weight: Weight given to sentiment analysis (0-1)
                              Higher values give more importance to sentiment
        """
        super().__init__()
        self.sentiment_weight = sentiment_weight
        self.technical_indicators = TechnicalIndicators()
        
        # Thresholds for technical indicators
        self.thresholds = {
            "RSI_OVERBOUGHT": 70,
            "RSI_OVERSOLD": 30,
            "MACD_SIGNAL": 0,
            "BB_UPPER_TOUCH": 0.95,  # Price touches 95% of the way to upper band
            "BB_LOWER_TOUCH": 0.95,  # Price touches 95% of the way to lower band
        }
    
    def generate_signals(self, market_data: List[Dict[str, float]], 
                         sentiment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate trading signals based on technical indicators and sentiment.
        
        Args:
            market_data: List of OHLCV data points
            sentiment_data: Optional sentiment data
            
        Returns:
            Dictionary of signals
        """
        if not market_data or len(market_data) < 30:
            return {"error": "Insufficient market data"}
        
        # Extract price data
        prices = np.array([candle['close'] for candle in market_data])
        highs = np.array([candle['high'] for candle in market_data])
        lows = np.array([candle['low'] for candle in market_data])
        volumes = np.array([candle['volume'] for candle in market_data])
        
        # Calculate technical indicators
        signals = {}
        
        # RSI
        rsi = self.technical_indicators.calculate_rsi(prices)
        signals["RSI"] = rsi[-1] if rsi is not None else None
        signals["RSI_SIGNAL"] = "oversold" if rsi is not None and rsi[-1] < self.thresholds["RSI_OVERSOLD"] else \
                              "overbought" if rsi is not None and rsi[-1] > self.thresholds["RSI_OVERBOUGHT"] else "neutral"
        
        # MACD
        macd, signal, hist = self.technical_indicators.calculate_macd(prices)
        signals["MACD"] = macd[-1] if macd is not None else None
        signals["MACD_SIGNAL"] = signal[-1] if signal is not None else None
        signals["MACD_HIST"] = hist[-1] if hist is not None else None
        signals["MACD_CROSSOVER"] = "bullish" if hist is not None and hist[-2] < 0 and hist[-1] > 0 else \
                                  "bearish" if hist is not None and hist[-2] > 0 and hist[-1] < 0 else "neutral"
        
        # Moving Averages
        ma_50 = self.technical_indicators.calculate_sma(prices, 50)
        ma_200 = self.technical_indicators.calculate_sma(prices, 200)
        signals["MA_50"] = ma_50[-1] if ma_50 is not None else None
        signals["MA_200"] = ma_200[-1] if ma_200 is not None else None
        
        # MA Crossover
        if ma_50 is not None and ma_200 is not None and len(ma_50) > 1 and len(ma_200) > 1:
            signals["MA_CROSSOVER"] = "bullish" if ma_50[-2] < ma_200[-2] and ma_50[-1] > ma_200[-1] else \
                                    "bearish" if ma_50[-2] > ma_200[-2] and ma_50[-1] < ma_200[-1] else "neutral"
        else:
            signals["MA_CROSSOVER"] = "neutral"
        
        # Bollinger Bands
        upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(prices)
        if upper is not None and middle is not None and lower is not None:
            signals["BB_UPPER"] = upper[-1]
            signals["BB_MIDDLE"] = middle[-1]
            signals["BB_LOWER"] = lower[-1]
            
            # Check if price is near the bands
            current_price = prices[-1]
            band_width = upper[-1] - lower[-1]
            upper_distance = (upper[-1] - current_price) / band_width if band_width > 0 else 0
            lower_distance = (current_price - lower[-1]) / band_width if band_width > 0 else 0
            
            signals["BB_POSITION"] = "upper" if upper_distance < 0.05 else \
                                   "lower" if lower_distance < 0.05 else "middle"
        
        # ATR (Volatility)
        atr = self.technical_indicators.calculate_atr(highs, lows, prices)
        signals["ATR"] = atr[-1] if atr is not None else None
        
        # Volume analysis
        avg_volume = np.mean(volumes[-10:])
        signals["VOLUME_RATIO"] = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Sentiment analysis
        if sentiment_data:
            signals["SENTIMENT_SCORE"] = sentiment_data.get("score", 0)
            signals["SENTIMENT_MAGNITUDE"] = sentiment_data.get("magnitude", 0)
            signals["SENTIMENT_TREND"] = sentiment_data.get("trend", "neutral")
            
            # News sentiment
            signals["NEWS_SENTIMENT"] = sentiment_data.get("news_sentiment", 0)
            
            # Social media sentiment
            signals["SOCIAL_SENTIMENT"] = sentiment_data.get("social_sentiment", 0)
        
        return signals
    
    def make_decision(self, signals: Dict[str, Any], risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """
        Make a trading decision based on signals and risk tolerance.
        
        Args:
            signals: Dictionary of trading signals
            risk_tolerance: Risk tolerance (0-1)
            
        Returns:
            Dictionary with trading decision
        """
        if "error" in signals:
            return {"action": "hold", "confidence": 0, "reason": signals["error"]}
        
        # Calculate technical score (-1 to 1)
        technical_score = self._calculate_technical_score(signals)
        
        # Calculate sentiment score (-1 to 1) if available
        sentiment_score = self._calculate_sentiment_score(signals)
        
        # Combine scores based on sentiment weight
        if sentiment_score is not None:
            combined_score = (1 - self.sentiment_weight) * technical_score + self.sentiment_weight * sentiment_score
        else:
            combined_score = technical_score
        
        # Adjust with risk tolerance
        # Higher risk tolerance means acting on smaller signal strengths
        threshold = 0.3 * (1 - risk_tolerance)  # Threshold decreases as risk_tolerance increases
        
        # Make decision
        action = "hold"
        confidence = abs(combined_score)
        reason = ""
        
        if combined_score > threshold:
            action = "buy"
            reason = f"Bullish signal: technical={technical_score:.2f}"
            if sentiment_score is not None:
                reason += f", sentiment={sentiment_score:.2f}"
        elif combined_score < -threshold:
            action = "sell"
            reason = f"Bearish signal: technical={technical_score:.2f}"
            if sentiment_score is not None:
                reason += f", sentiment={sentiment_score:.2f}"
        else:
            reason = "Signal strength below threshold"
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "technical_score": technical_score,
            "sentiment_score": sentiment_score,
            "combined_score": combined_score,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_technical_score(self, signals: Dict[str, Any]) -> float:
        """
        Calculate a technical score from -1 (bearish) to 1 (bullish).
        
        Args:
            signals: Dictionary of trading signals
            
        Returns:
            Technical score from -1 to 1
        """
        score = 0.0
        components = 0
        
        # RSI component
        if "RSI" in signals and signals["RSI"] is not None:
            rsi = signals["RSI"]
            if rsi < 30:
                # Oversold - bullish
                score += 0.5 + (30 - rsi) / 60  # Max contribution: 1.0 at RSI=0
            elif rsi > 70:
                # Overbought - bearish
                score -= 0.5 + (rsi - 70) / 60  # Max contribution: -1.0 at RSI=100
            else:
                # Neutral with slight bias
                score += (50 - rsi) / 100  # Range: -0.2 to 0.2
            components += 1
        
        # MACD component
        if "MACD_CROSSOVER" in signals:
            if signals["MACD_CROSSOVER"] == "bullish":
                score += 0.75
            elif signals["MACD_CROSSOVER"] == "bearish":
                score -= 0.75
            components += 1
        
        # MA Crossover component
        if "MA_CROSSOVER" in signals:
            if signals["MA_CROSSOVER"] == "bullish":
                score += 1.0
            elif signals["MA_CROSSOVER"] == "bearish":
                score -= 1.0
            components += 1
        
        # Price relative to MAs
        if "MA_50" in signals and "MA_200" in signals and signals["MA_50"] and signals["MA_200"]:
            current_price = signals.get("current_price", 0)
            if current_price > 0:
                # Price above both MAs - bullish
                if current_price > signals["MA_50"] and current_price > signals["MA_200"]:
                    score += 0.5
                # Price below both MAs - bearish
                elif current_price < signals["MA_50"] and current_price < signals["MA_200"]:
                    score -= 0.5
            components += 1
        
        # Bollinger Bands
        if "BB_POSITION" in signals:
            if signals["BB_POSITION"] == "lower":
                score += 0.5  # Near lower band - potential reversal up
            elif signals["BB_POSITION"] == "upper":
                score -= 0.5  # Near upper band - potential reversal down
            components += 1
        
        # Volume confirmation
        if "VOLUME_RATIO" in signals:
            volume_ratio = signals["VOLUME_RATIO"]
            # High volume confirms the direction
            if volume_ratio > 1.5:
                # Amplify the existing signal
                score *= 1.2
            components += 1
        
        # Normalize the score to -1 to 1 range
        if components > 0:
            normalized_score = score / components
            # Clamp between -1 and 1
            return max(-1.0, min(1.0, normalized_score))
        else:
            return 0.0
    
    def _calculate_sentiment_score(self, signals: Dict[str, Any]) -> Optional[float]:
        """
        Calculate a sentiment score from -1 (bearish) to 1 (bullish).
        
        Args:
            signals: Dictionary of trading signals
            
        Returns:
            Sentiment score from -1 to 1, or None if no sentiment data
        """
        if "SENTIMENT_SCORE" not in signals:
            return None
        
        sentiment_components = []
        
        # Overall sentiment score (-1 to 1)
        if "SENTIMENT_SCORE" in signals:
            sentiment_components.append(signals["SENTIMENT_SCORE"])
        
        # News sentiment (-1 to 1)
        if "NEWS_SENTIMENT" in signals:
            sentiment_components.append(signals["NEWS_SENTIMENT"])
        
        # Social sentiment (-1 to 1)
        if "SOCIAL_SENTIMENT" in signals:
            sentiment_components.append(signals["SOCIAL_SENTIMENT"])
        
        # Average the components
        if sentiment_components:
            avg_sentiment = sum(sentiment_components) / len(sentiment_components)
            
            # Apply a magnitude factor if available
            if "SENTIMENT_MAGNITUDE" in signals:
                magnitude = signals["SENTIMENT_MAGNITUDE"]
                # Scale magnitude to 0.5-1.5 range to avoid completely negating sentiment
                magnitude_factor = 0.5 + min(magnitude, 1.0)
                return avg_sentiment * magnitude_factor
            else:
                return avg_sentiment
        
        return None
    
    def adapt_from_reflection(self, reflection: Dict[str, Any]) -> None:
        """
        Adapt strategy parameters based on reflection insights.
        
        Args:
            reflection: Dictionary with reflection data
        """
        if not reflection:
            return
        
        # Extract insights
        insights = reflection.get("insights", [])
        win_rate = reflection.get("win_rate", 0.5)
        
        # Adjust sentiment weight based on performance
        if win_rate < 0.4:
            # Poor performance - try different approach
            if self.sentiment_weight > 0.5:
                # If sentiment was emphasized, reduce it
                self.sentiment_weight = max(0.1, self.sentiment_weight - 0.1)
            else:
                # If technical was emphasized, try more sentiment
                self.sentiment_weight = min(0.9, self.sentiment_weight + 0.1)
        elif win_rate > 0.6:
            # Good performance - keep current balance
            pass
        
        # Adjust thresholds based on insights
        for insight in insights:
            if "RSI" in insight and "successful" in insight:
                # Adjust RSI thresholds based on success
                if "oversold" in insight.lower():
                    self.thresholds["RSI_OVERSOLD"] = max(20, self.thresholds["RSI_OVERSOLD"] - 2)
                if "overbought" in insight.lower():
                    self.thresholds["RSI_OVERBOUGHT"] = min(80, self.thresholds["RSI_OVERBOUGHT"] + 2)
            
            if "MACD" in insight and "failed" in insight:
                # If MACD signals are failing, make them more conservative
                self.thresholds["MACD_SIGNAL"] = 0.001 if self.thresholds["MACD_SIGNAL"] == 0 else self.thresholds["MACD_SIGNAL"] * 1.2 