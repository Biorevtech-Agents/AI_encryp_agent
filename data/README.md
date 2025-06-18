# Data Directory Structure

This directory contains all the data used by the Encryption trading agent.

## Directory Structure

- `raw/`: Raw data files from various sources
  - Market data
  - News articles
  - Social media feeds
  
- `processed/`: Processed and cleaned data
  - Technical indicators
  - Sentiment analysis results
  - Feature engineered data
  
- `models/`: Saved model files
  - Trained models
  - Model checkpoints
  - Model configurations

## Data Sources

1. Market Data:
   - Binance API
   - Historical price data
   - Trading volume
   - Order book data

2. News and Sentiment:
   - News API
   - Social media feeds
   - Sentiment analysis results

## Data Processing

All raw data is processed and stored in the `processed/` directory. This includes:
- Technical indicators calculation
- Sentiment analysis
- Feature engineering
- Data normalization

## Model Storage

Trained models and their configurations are stored in the `models/` directory.
Each model version should include:
- Model weights
- Configuration file
- Performance metrics
- Training history 