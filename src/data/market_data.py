import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union

from src.config.config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_PAIRS, DEFAULT_TIMEFRAME, LOOKBACK_PERIOD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_data')

class MarketDataFetcher:
    """Class to fetch and process market data from cryptocurrency exchanges."""
    
    def __init__(self):
        """Initialize the market data fetcher with exchange connections."""
        self.exchanges = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize connections to cryptocurrency exchanges."""
        try:
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_API_SECRET,
                'enableRateLimit': True,
            })
            logger.info("Successfully connected to Binance exchange")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            # Continue without API keys - can still fetch public data
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
            })
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, 
                   limit: int = LOOKBACK_PERIOD) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '15m', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            exchange = self.exchanges['binance']
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} {timeframe} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Fetch order book data for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Depth of the order book to fetch
            
        Returns:
            Dictionary containing bids and asks
        """
        try:
            exchange = self.exchanges['binance']
            order_book = exchange.fetch_order_book(symbol, limit)
            logger.info(f"Successfully fetched order book for {symbol}")
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing ticker information
        """
        try:
            exchange = self.exchanges['binance']
            ticker = exchange.fetch_ticker(symbol)
            logger.info(f"Successfully fetched ticker for {symbol}")
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    def fetch_all_pairs_data(self, timeframe: str = DEFAULT_TIMEFRAME) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all configured trading pairs.
        
        Args:
            timeframe: Candle timeframe
            
        Returns:
            Dictionary mapping symbols to their OHLCV DataFrames
        """
        all_data = {}
        for symbol in TRADING_PAIRS:
            df = self.fetch_ohlcv(symbol, timeframe)
            if not df.empty:
                all_data[symbol] = df
            # Rate limiting
            time.sleep(1)
        return all_data
    
    def get_market_overview(self) -> Dict:
        """
        Get an overview of current market conditions.
        
        Returns:
            Dictionary with market overview data
        """
        overview = {
            'timestamp': datetime.now(),
            'tickers': {},
            'daily_changes': {},
            'volumes': {},
        }
        
        for symbol in TRADING_PAIRS:
            ticker = self.fetch_ticker(symbol)
            if ticker:
                overview['tickers'][symbol] = ticker['last']
                overview['daily_changes'][symbol] = ticker['percentage']
                overview['volumes'][symbol] = ticker['quoteVolume']
        
        return overview 