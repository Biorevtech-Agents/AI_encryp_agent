import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from src.agent.base_agent import BaseAgent
from src.data.market_data import MarketDataFetcher
from src.data.news_sentiment import SentimentAnalyzer
from src.strategies.base_strategy import BaseStrategy
from src.strategies.technical_strategy import TechnicalStrategy
from src.config.config import (
    TRADING_PAIRS, DEFAULT_TIMEFRAME, LOOKBACK_PERIOD, 
    MAX_POSITION_SIZE, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE,
    DATA_UPDATE_INTERVAL
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TradingAgent")

class TradingAgent(BaseAgent):
    """
    Autonomous trading agent that makes trading decisions based on
    technical analysis and sentiment data.
    """
    def __init__(
        self,
        strategy: BaseStrategy,
        market_data: MarketDataFetcher,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        trading_pairs: List[str] = TRADING_PAIRS,
        timeframe: str = DEFAULT_TIMEFRAME,
        max_position_size: float = MAX_POSITION_SIZE,
        stop_loss_pct: float = STOP_LOSS_PERCENTAGE,
        take_profit_pct: float = TAKE_PROFIT_PERCENTAGE,
        **kwargs
    ):
        super().__init__(strategy, market_data, sentiment_analyzer, **kwargs)
        
        self.trading_pairs = trading_pairs
        self.timeframe = timeframe
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Track active orders and positions
        self.active_orders = {}
        self.stop_losses = {}
        self.take_profits = {}
        
        # Last data update time
        self.last_data_update = time.time()
        
        logger.info(f"Trading agent initialized with {len(trading_pairs)} trading pairs")
    
    def run(self, iterations: Optional[int] = None) -> None:
        """
        Run the trading agent for a specified number of iterations or indefinitely
        
        Args:
            iterations: Number of iterations to run. If None, run indefinitely.
        """
        iteration_count = 0
        
        logger.info("Starting trading agent...")
        
        try:
            while iterations is None or iteration_count < iterations:
                # Update market data if needed
                self._update_data_if_needed()
                
                # Check for completed trades
                self._check_completed_trades()
                
                # Check stop losses and take profits
                self._check_stop_losses_and_take_profits()
                
                # Make trading decisions
                self._make_trading_decisions()
                
                # Reflect on performance periodically
                if (time.time() - self.last_reflection_time > self.reflection_frequency * 3600 and 
                    self.trade_count >= self.reflection_frequency):
                    reflection = self.reflect()
                    logger.info(f"Agent reflection completed: Win rate {reflection.get('win_rate', 0):.2f}")
                    
                    # Apply suggested adjustments
                    self._apply_reflection_adjustments(reflection)
                
                # Sleep to avoid API rate limits
                time.sleep(5)
                
                if iterations is not None:
                    iteration_count += 1
                    
        except KeyboardInterrupt:
            logger.info("Trading agent stopped by user")
        except Exception as e:
            logger.error(f"Error in trading agent: {str(e)}", exc_info=True)
        finally:
            # Save state before exiting
            self.save_state("agent_state.json")
            logger.info("Trading agent state saved")
    
    def _update_data_if_needed(self) -> None:
        """Update market data and sentiment if enough time has passed"""
        current_time = time.time()
        
        # Check if we need to update data
        if current_time - self.last_data_update >= DATA_UPDATE_INTERVAL:
            logger.info("Updating market data...")
            
            # Update market data for all trading pairs
            for pair in self.trading_pairs:
                try:
                    self.market_data.fetch_ohlcv(pair, self.timeframe, LOOKBACK_PERIOD)
                    logger.debug(f"Updated market data for {pair}")
                except Exception as e:
                    logger.error(f"Failed to update market data for {pair}: {str(e)}")
            
            # Update sentiment data if available
            if self.sentiment_analyzer:
                try:
                    for pair in self.trading_pairs:
                        # Extract base currency from pair (e.g., "BTC" from "BTC/USDT")
                        base_currency = pair.split('/')[0]
                        self.sentiment_analyzer.update_sentiment(base_currency)
                        logger.debug(f"Updated sentiment for {base_currency}")
                except Exception as e:
                    logger.error(f"Failed to update sentiment data: {str(e)}")
            
            self.last_data_update = current_time
    
    def _check_completed_trades(self) -> None:
        """Check for completed trades and update portfolio"""
        # In a real implementation, this would check with the exchange API
        # For now, we'll simulate completed trades
        completed_orders = []
        
        for order_id, order in self.active_orders.items():
            symbol = order["symbol"]
            price = self.market_data.get_current_price(symbol)
            
            if not price:
                continue
                
            # Simulate order execution (in a real system, check exchange API)
            if order["type"] == "limit":
                # For limit orders, check if price crossed the limit price
                if (order["side"] == "buy" and price <= order["price"]) or \
                   (order["side"] == "sell" and price >= order["price"]):
                    # Order executed
                    trade_result = {
                        "symbol": symbol,
                        "action": order["side"],
                        "amount": order["amount"],
                        "price": order["price"],
                        "timestamp": datetime.now().isoformat(),
                        "order_id": order_id
                    }
                    
                    # Update portfolio
                    self.update_portfolio(trade_result)
                    
                    # Add to memory
                    self.update_memory("trades", trade_result)
                    
                    # Mark order as completed
                    completed_orders.append(order_id)
                    
                    logger.info(f"Order executed: {order['side']} {order['amount']} {symbol} at {order['price']}")
            
            elif order["type"] == "market":
                # Market orders execute immediately
                trade_result = {
                    "symbol": symbol,
                    "action": order["side"],
                    "amount": order["amount"],
                    "price": price,  # Use current price
                    "timestamp": datetime.now().isoformat(),
                    "order_id": order_id
                }
                
                # Update portfolio
                self.update_portfolio(trade_result)
                
                # Add to memory
                self.update_memory("trades", trade_result)
                
                # Mark order as completed
                completed_orders.append(order_id)
                
                logger.info(f"Market order executed: {order['side']} {order['amount']} {symbol} at {price}")
        
        # Remove completed orders
        for order_id in completed_orders:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
    
    def _check_stop_losses_and_take_profits(self) -> None:
        """Check if any stop losses or take profits have been triggered"""
        # Check each position for stop loss or take profit
        for symbol in list(self.portfolio["positions"].keys()):
            current_price = self.market_data.get_current_price(symbol)
            if not current_price:
                continue
                
            position = self.portfolio["positions"][symbol]
            avg_price = position["avg_price"]
            
            # Check stop loss
            if symbol in self.stop_losses:
                stop_price = self.stop_losses[symbol]
                if current_price <= stop_price:
                    # Stop loss triggered
                    self._execute_market_order(symbol, "sell", position["amount"])
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    continue
            
            # Check take profit
            if symbol in self.take_profits:
                take_profit_price = self.take_profits[symbol]
                if current_price >= take_profit_price:
                    # Take profit triggered
                    self._execute_market_order(symbol, "sell", position["amount"])
                    logger.info(f"Take profit triggered for {symbol} at {current_price}")
    
    def _make_trading_decisions(self) -> None:
        """Analyze market data and make trading decisions"""
        for pair in self.trading_pairs:
            try:
                # Skip if we already have a position in this pair
                if pair in self.portfolio["positions"]:
                    continue
                    
                # Get market data
                ohlcv_data = self.market_data.get_ohlcv_data(pair, self.timeframe)
                if ohlcv_data is None or len(ohlcv_data) < LOOKBACK_PERIOD:
                    logger.warning(f"Insufficient data for {pair}, skipping")
                    continue
                
                # Get sentiment data if available
                sentiment_data = None
                if self.sentiment_analyzer:
                    base_currency = pair.split('/')[0]
                    sentiment_data = self.sentiment_analyzer.get_sentiment(base_currency)
                
                # Get trading signals from strategy
                signals = self.strategy.generate_signals(ohlcv_data, sentiment_data)
                
                # Record market observation
                self.update_memory("market_observations", {
                    "symbol": pair,
                    "timestamp": datetime.now().isoformat(),
                    "signals": signals,
                    "current_price": ohlcv_data[-1]['close']
                })
                
                # Make decision based on signals
                decision = self.strategy.make_decision(signals, self.risk_tolerance)
                
                if decision["action"] == "buy":
                    # Calculate position size based on portfolio and risk
                    max_amount = self.portfolio["cash"] * self.max_position_size
                    price = ohlcv_data[-1]['close']
                    amount = max_amount / price
                    
                    # Execute order
                    self._execute_limit_order(pair, "buy", amount, price)
                    
                    # Set stop loss and take profit
                    self.stop_losses[pair] = price * (1 - self.stop_loss_pct)
                    self.take_profits[pair] = price * (1 + self.take_profit_pct)
                    
                    logger.info(f"BUY signal for {pair} at {price}. Amount: {amount:.6f}")
                
            except Exception as e:
                logger.error(f"Error making trading decision for {pair}: {str(e)}")
    
    def _execute_limit_order(self, symbol: str, side: str, amount: float, price: float) -> str:
        """
        Execute a limit order
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Amount to buy/sell
            price: Limit price
            
        Returns:
            order_id: ID of the placed order
        """
        # In a real implementation, this would call the exchange API
        order_id = f"order_{int(time.time())}_{symbol}_{side}"
        
        order = {
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "amount": amount,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
        
        self.active_orders[order_id] = order
        
        # If it's a sell order, update the stop loss and take profit
        if side == "sell" and symbol in self.stop_losses:
            del self.stop_losses[symbol]
        
        if side == "sell" and symbol in self.take_profits:
            del self.take_profits[symbol]
        
        return order_id
    
    def _execute_market_order(self, symbol: str, side: str, amount: float) -> str:
        """
        Execute a market order
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Amount to buy/sell
            
        Returns:
            order_id: ID of the placed order
        """
        # In a real implementation, this would call the exchange API
        order_id = f"order_{int(time.time())}_{symbol}_{side}"
        
        price = self.market_data.get_current_price(symbol)
        if not price:
            logger.error(f"Could not get current price for {symbol}")
            return ""
        
        order = {
            "symbol": symbol,
            "side": side,
            "type": "market",
            "amount": amount,
            "price": price,  # Estimated price
            "timestamp": datetime.now().isoformat()
        }
        
        self.active_orders[order_id] = order
        
        # If it's a sell order, update the stop loss and take profit
        if side == "sell" and symbol in self.stop_losses:
            del self.stop_losses[symbol]
        
        if side == "sell" and symbol in self.take_profits:
            del self.take_profits[symbol]
        
        return order_id
    
    def _apply_reflection_adjustments(self, reflection: Dict[str, Any]) -> None:
        """Apply adjustments suggested by reflection"""
        if not reflection:
            return
            
        adjustments = reflection.get("strategy_adjustments", {})
        
        # Adjust risk tolerance
        if "suggested_risk" in adjustments:
            old_risk = self.risk_tolerance
            self.risk_tolerance = adjustments["suggested_risk"]
            logger.info(f"Adjusted risk tolerance from {old_risk:.2f} to {self.risk_tolerance:.2f}")
        
        # Let the strategy adapt based on reflection
        if hasattr(self.strategy, "adapt_from_reflection"):
            self.strategy.adapt_from_reflection(reflection)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of agent's performance"""
        return {
            "total_trades": self.performance_metrics["total_trades"],
            "successful_trades": self.performance_metrics["successful_trades"],
            "failed_trades": self.performance_metrics["failed_trades"],
            "win_rate": self.performance_metrics["win_rate"],
            "profit_loss": self.performance_metrics["profit_loss"],
            "portfolio_value": self.portfolio["total_value"],
            "return_pct": (self.portfolio["total_value"] / self.portfolio_size - 1) * 100
        } 