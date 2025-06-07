#!/usr/bin/env python3
"""
Autonomous Crypto Trading Agent

This script initializes and runs the autonomous trading agent.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.trading_agent import TradingAgent
from src.data.market_data import MarketDataFetcher
from src.data.news_sentiment import SentimentAnalyzer
from src.strategies.technical_strategy import TechnicalStrategy
from src.config.config import (
    TRADING_PAIRS, DEFAULT_TIMEFRAME, LOOKBACK_PERIOD,
    PORTFOLIO_SIZE, RISK_TOLERANCE, TECHNICAL_INDICATORS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the autonomous crypto trading agent')
    
    parser.add_argument('--pairs', nargs='+', default=TRADING_PAIRS,
                        help=f'Trading pairs to use (default: {TRADING_PAIRS})')
    
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                        help=f'Timeframe for analysis (default: {DEFAULT_TIMEFRAME})')
    
    parser.add_argument('--portfolio', type=float, default=PORTFOLIO_SIZE,
                        help=f'Initial portfolio size in USDT (default: {PORTFOLIO_SIZE})')
    
    parser.add_argument('--risk', type=float, default=RISK_TOLERANCE,
                        help=f'Risk tolerance (0-1) (default: {RISK_TOLERANCE})')
    
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode instead of live trading')
    
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of iterations to run (default: run indefinitely)')
    
    parser.add_argument('--no-sentiment', action='store_true',
                        help='Disable sentiment analysis')
    
    return parser.parse_args()

def main():
    """Main function to run the trading agent"""
    args = parse_arguments()
    
    try:
        logger.info("Initializing trading components...")
        
        # Initialize market data fetcher
        market_data = MarketDataFetcher()
        
        # Initialize sentiment analyzer if enabled
        sentiment_analyzer = None
        if not args.no_sentiment:
            sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize technical strategy
        strategy = TechnicalStrategy()
        
        # Initialize trading agent
        agent = TradingAgent(
            strategy=strategy,
            market_data=market_data,
            sentiment_analyzer=sentiment_analyzer,
            trading_pairs=args.pairs,
            timeframe=args.timeframe,
            portfolio_size=args.portfolio,
            risk_tolerance=args.risk
        )
        
        logger.info(f"Trading agent initialized with {len(args.pairs)} pairs")
        
        # Run the agent
        if args.backtest:
            logger.info("Running in backtest mode")
            # TODO: Implement backtest mode
            raise NotImplementedError("Backtest mode not yet implemented")
        else:
            logger.info("Running in live trading mode")
            agent.run(iterations=args.iterations)
        
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading session ended")

if __name__ == "__main__":
    main() 