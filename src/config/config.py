import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Trading Parameters
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
DEFAULT_TIMEFRAME = '1h'  # 1 hour candles
LOOKBACK_PERIOD = 30  # Number of candles to analyze
PORTFOLIO_SIZE = 1000  # Initial portfolio size in USDT
MAX_POSITION_SIZE = 0.2  # Maximum position size as percentage of portfolio
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.1  # 10% take profit

# Agent Parameters
RISK_TOLERANCE = 0.5  # 0-1 scale, higher means more risk-taking
MEMORY_DECAY_RATE = 0.9  # Rate at which old memories lose importance
REFLECTION_FREQUENCY = 10  # Reflect after every N trades

# Data Parameters
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'MA_50', 'MA_200', 'BOLLINGER_BANDS', 'ATR'
]
SENTIMENT_SOURCES = ['news', 'twitter', 'reddit']
DATA_UPDATE_INTERVAL = 3600  # Update data every hour (in seconds)

# Model Parameters
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english' 