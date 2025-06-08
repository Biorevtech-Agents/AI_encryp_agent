# Crypto Trading Agent

[![Build Status](https://github.com/Biorevtech-Agents/crypto_trading_agent/workflows/CI%2FCD/badge.svg)](https://github.com/Biorevtech-Agents/crypto_trading_agent/actions)

An AI-powered cryptocurrency trading agent that uses advanced machine learning for automated trading.

## System Requirements

- **Python**: Version 3.12 (required)
  - The project is specifically built and tested with Python 3.12
  - Other versions are not officially supported
- Docker and Docker Compose (for containerized deployment)
- Linux/Unix environment recommended

## Features

- Autonomous trading with multiple strategies
- Technical analysis using various indicators (RSI, MACD, Moving Averages, etc.)
- Sentiment analysis from news and social media
- Self-reflection and strategy adaptation
- Risk management with dynamic position sizing
- Real-time monitoring with Prometheus and Grafana
- Comprehensive test suite with unit and integration tests

## Prerequisites

Before you begin, ensure you have:
- Python 3.12 installed (higher versions are not officially supported)
- Docker and Docker Compose (for containerized deployment)
- Binance API credentials
- News API key

## Installation

1. Ensure you have Python 3.12:
```bash
python --version  # Should output Python 3.12.x
```

2. Clone the repository:
```bash
git clone https://github.com/Biorevtech-Agents/crypto_trading_agent.git
cd crypto_trading_agent
```

3. Create and activate a virtual environment with Python 3.12:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Copy the example environment file and update with your credentials:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Set up pre-commit hooks:
```bash
make install
```

3. Run tests:
```bash
make test  # Run all tests
make test-unit  # Run only unit tests
make test-integration  # Run only integration tests
```

4. Run linting and formatting:
```bash
make lint  # Run linters
make format  # Format code
```

## Running the Agent

### Local Development

1. Start the agent in development mode:
```bash
python -m src.main
```

2. Start with specific configuration:
```bash
python -m src.main --config path/to/config.yml
```

### Docker Deployment

1. Build and start the containers:
```bash
docker-compose up -d
```

2. View logs:
```bash
docker-compose logs -f trading_agent
```

3. Stop the containers:
```bash
docker-compose down
```

## Monitoring

The agent comes with built-in monitoring using Prometheus and Grafana.

1. Access Grafana dashboard:
   - URL: http://localhost:3000
   - Default credentials: admin/secret

2. Available metrics:
   - Portfolio value
   - Trade win rate
   - Total profit/loss
   - Active trades
   - Technical indicators
   - Sentiment scores

## Configuration

The agent can be configured through environment variables or a configuration file:

- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `NEWS_API_KEY`: Your News API key
- `RISK_TOLERANCE`: Risk tolerance (0-1)
- `PORTFOLIO_SIZE`: Initial portfolio size in USDT
- `MEMORY_DECAY_RATE`: Rate at which old memories lose importance
- See `.env.example` for all available options

## Project Structure

```
crypto_trading_agent/
├── src/
│   ├── agent/          # Trading agent implementation
│   ├── config/         # Configuration management
│   ├── data/           # Data handling and storage
│   ├── models/         # ML models and predictions
│   ├── strategies/     # Trading strategies
│   └── utils/          # Utility functions
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── monitoring/
│   ├── grafana/       # Grafana configuration
│   └── prometheus/    # Prometheus configuration
└── docker-compose.yml # Container orchestration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

Please ensure your code:
- Passes all tests
- Follows the project's coding style
- Includes appropriate documentation
- Adds tests for new features

## Warning

This is an experimental trading bot. Use at your own risk. Never trade with money you cannot afford to lose.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 