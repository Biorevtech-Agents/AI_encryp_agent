import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.data.market_data import MarketDataFetcher
from src.data.news_sentiment import SentimentAnalyzer
from src.strategies.base_strategy import BaseStrategy
from src.utils.technical_indicators import TechnicalIndicators
from src.config.config import (
    TRADING_PAIRS, DEFAULT_TIMEFRAME, LOOKBACK_PERIOD, 
    PORTFOLIO_SIZE, RISK_TOLERANCE, MEMORY_DECAY_RATE,
    REFLECTION_FREQUENCY
)

class BaseAgent:
    """
    Base class for autonomous trading agents.
    Handles memory, reflection, and decision-making processes.
    """
    def __init__(
        self,
        strategy: BaseStrategy,
        market_data: MarketDataFetcher,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        portfolio_size: float = PORTFOLIO_SIZE,
        risk_tolerance: float = RISK_TOLERANCE,
        memory_decay_rate: float = MEMORY_DECAY_RATE,
        reflection_frequency: int = REFLECTION_FREQUENCY
    ):
        self.strategy = strategy
        self.market_data = market_data
        self.sentiment_analyzer = sentiment_analyzer
        self.portfolio_size = portfolio_size
        self.risk_tolerance = risk_tolerance
        self.memory_decay_rate = memory_decay_rate
        self.reflection_frequency = reflection_frequency
        
        # Initialize memory
        self.memory = {
            "trades": [],
            "reflections": [],
            "market_observations": [],
            "portfolio_history": []
        }
        
        # Initialize portfolio
        self.portfolio = {
            "cash": portfolio_size,
            "positions": {},
            "total_value": portfolio_size
        }
        
        # Track performance
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "profit_loss": 0.0,
            "win_rate": 0.0
        }
        
        # Trading state
        self.trade_count = 0
        self.last_reflection_time = time.time()
    
    def update_memory(self, memory_type: str, data: Dict[str, Any]) -> None:
        """
        Update agent's memory with new information
        
        Args:
            memory_type: Type of memory to update (trades, reflections, etc.)
            data: The data to add to memory
        """
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Add to memory
        if memory_type in self.memory:
            self.memory[memory_type].append(data)
        else:
            self.memory[memory_type] = [data]
        
        # Apply memory decay to older memories
        self._apply_memory_decay(memory_type)
    
    def _apply_memory_decay(self, memory_type: str) -> None:
        """Apply memory decay to older memories to prioritize recent information"""
        if memory_type not in self.memory or not self.memory[memory_type]:
            return
        
        for i, memory_item in enumerate(self.memory[memory_type]):
            # Skip the most recent memory
            if i == len(self.memory[memory_type]) - 1:
                continue
            
            # Calculate age factor based on position in memory
            age_factor = (len(self.memory[memory_type]) - i - 1) / len(self.memory[memory_type])
            
            # Apply decay
            if "importance" in memory_item:
                memory_item["importance"] *= (self.memory_decay_rate ** age_factor)
    
    def reflect(self) -> Dict[str, Any]:
        """
        Reflect on past trades and market observations to improve strategy
        
        Returns:
            Dict containing reflection insights
        """
        # Skip if not enough trades
        if self.trade_count < self.reflection_frequency:
            return {}
        
        # Analyze past trades
        wins = sum(1 for trade in self.memory["trades"][-self.reflection_frequency:] 
                 if trade.get("profit_loss", 0) > 0)
        losses = self.reflection_frequency - wins
        
        # Calculate win rate
        win_rate = wins / self.reflection_frequency if self.reflection_frequency > 0 else 0
        
        # Generate insights
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "period_trades": self.reflection_frequency,
            "win_rate": win_rate,
            "insights": self._generate_insights(),
            "strategy_adjustments": self._suggest_strategy_adjustments(win_rate)
        }
        
        # Update memory with reflection
        self.update_memory("reflections", reflection)
        self.last_reflection_time = time.time()
        
        return reflection
    
    def _generate_insights(self) -> List[str]:
        """Generate insights based on trading history"""
        insights = []
        
        # Analyze recent trades
        recent_trades = self.memory["trades"][-self.reflection_frequency:]
        if not recent_trades:
            return ["Insufficient trading data for insights"]
        
        # Check for patterns in successful trades
        successful_trades = [t for t in recent_trades if t.get("profit_loss", 0) > 0]
        if successful_trades:
            # Look for common indicators in successful trades
            common_indicators = self._find_common_indicators(successful_trades)
            if common_indicators:
                insights.append(f"Successful trades often involve these indicators: {', '.join(common_indicators)}")
        
        # Check for patterns in failed trades
        failed_trades = [t for t in recent_trades if t.get("profit_loss", 0) <= 0]
        if failed_trades:
            # Look for common indicators in failed trades
            common_indicators = self._find_common_indicators(failed_trades)
            if common_indicators:
                insights.append(f"Failed trades often involve these indicators: {', '.join(common_indicators)}")
        
        return insights
    
    def _find_common_indicators(self, trades: List[Dict[str, Any]]) -> List[str]:
        """Find common technical indicators in a set of trades"""
        if not trades:
            return []
        
        indicator_counts = {}
        for trade in trades:
            signals = trade.get("signals", {})
            for indicator, value in signals.items():
                if indicator not in indicator_counts:
                    indicator_counts[indicator] = 0
                indicator_counts[indicator] += 1
        
        # Return indicators that appear in at least 50% of trades
        threshold = len(trades) * 0.5
        return [ind for ind, count in indicator_counts.items() if count >= threshold]
    
    def _suggest_strategy_adjustments(self, win_rate: float) -> Dict[str, Any]:
        """Suggest adjustments to strategy based on performance"""
        adjustments = {}
        
        # Adjust risk based on win rate
        if win_rate < 0.4:
            adjustments["risk_adjustment"] = "decrease"
            adjustments["suggested_risk"] = max(0.1, self.risk_tolerance * 0.8)
        elif win_rate > 0.6:
            adjustments["risk_adjustment"] = "increase"
            adjustments["suggested_risk"] = min(1.0, self.risk_tolerance * 1.2)
        else:
            adjustments["risk_adjustment"] = "maintain"
            adjustments["suggested_risk"] = self.risk_tolerance
        
        return adjustments
    
    def update_portfolio(self, trade_result: Dict[str, Any]) -> None:
        """
        Update portfolio based on trade results
        
        Args:
            trade_result: Dictionary containing trade details
        """
        symbol = trade_result.get("symbol")
        action = trade_result.get("action")
        amount = trade_result.get("amount", 0)
        price = trade_result.get("price", 0)
        
        if not all([symbol, action, amount, price]):
            return
        
        # Update positions and cash
        if action == "buy":
            # Deduct cash
            cost = amount * price
            if cost > self.portfolio["cash"]:
                # Not enough cash
                return
            
            self.portfolio["cash"] -= cost
            
            # Add to positions
            if symbol in self.portfolio["positions"]:
                # Average down/up
                current_amount = self.portfolio["positions"][symbol]["amount"]
                current_price = self.portfolio["positions"][symbol]["avg_price"]
                
                # Calculate new average price
                total_amount = current_amount + amount
                new_avg_price = ((current_amount * current_price) + (amount * price)) / total_amount
                
                self.portfolio["positions"][symbol] = {
                    "amount": total_amount,
                    "avg_price": new_avg_price
                }
            else:
                # New position
                self.portfolio["positions"][symbol] = {
                    "amount": amount,
                    "avg_price": price
                }
        
        elif action == "sell":
            # Check if we have the position
            if symbol not in self.portfolio["positions"]:
                return
            
            # Check if we have enough to sell
            current_amount = self.portfolio["positions"][symbol]["amount"]
            if amount > current_amount:
                amount = current_amount  # Sell what we have
            
            # Calculate profit/loss
            avg_price = self.portfolio["positions"][symbol]["avg_price"]
            profit_loss = (price - avg_price) * amount
            
            # Update cash
            self.portfolio["cash"] += amount * price
            
            # Update position
            if amount == current_amount:
                # Sold entire position
                del self.portfolio["positions"][symbol]
            else:
                # Partial sale
                self.portfolio["positions"][symbol]["amount"] -= amount
            
            # Update performance metrics
            self.performance_metrics["profit_loss"] += profit_loss
            self.performance_metrics["total_trades"] += 1
            if profit_loss > 0:
                self.performance_metrics["successful_trades"] += 1
            else:
                self.performance_metrics["failed_trades"] += 1
            
            # Calculate win rate
            total = self.performance_metrics["total_trades"]
            wins = self.performance_metrics["successful_trades"]
            self.performance_metrics["win_rate"] = wins / total if total > 0 else 0
        
        # Update total portfolio value
        self._update_portfolio_value()
        
        # Record portfolio history
        self.update_memory("portfolio_history", {
            "timestamp": datetime.now().isoformat(),
            "portfolio": self.portfolio.copy(),
            "performance": self.performance_metrics.copy()
        })
    
    def _update_portfolio_value(self) -> None:
        """Update the total portfolio value based on current prices"""
        total_value = self.portfolio["cash"]
        
        # Get current prices for all positions
        for symbol, position in self.portfolio["positions"].items():
            # Get current price
            current_price = self.market_data.get_current_price(symbol)
            if current_price:
                position_value = position["amount"] * current_price
                total_value += position_value
        
        self.portfolio["total_value"] = total_value
    
    def save_state(self, filepath: str) -> None:
        """Save agent state to file"""
        state = {
            "memory": self.memory,
            "portfolio": self.portfolio,
            "performance_metrics": self.performance_metrics,
            "trade_count": self.trade_count,
            "last_reflection_time": self.last_reflection_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load agent state from file"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.memory = state.get("memory", self.memory)
        self.portfolio = state.get("portfolio", self.portfolio)
        self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
        self.trade_count = state.get("trade_count", self.trade_count)
        self.last_reflection_time = state.get("last_reflection_time", self.last_reflection_time) 