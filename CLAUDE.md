# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency automated trading bot for BitFlyer FX that can operate with either dummy models or machine learning models trained in the companion `liza_simple_predictor` project. The system integrates technical analysis, price prediction, and automated order execution.

## Core Architecture

The system uses a modular design with clear separation between:
- **Trading Logic**: `trader.py` - Main trading orchestration and position management
- **Exchange API**: `exchanges.py` - BitFlyer API wrapper for orders and market data
- **Models**: `ml_model.py` and `ml_modules/` - ML model integration with fallback to dummy models
- **Historical Data**: `historical_data.py` - DynamoDB integration for instant startup

### Key Components

1. **TradingModel**: Core prediction engine that maintains price history and makes buy/sell decisions
2. **TraderManager**: Orchestrates the trading loop, API calls, and position management
3. **HybridTechnicalModel**: TensorFlow-based ML model wrapper with technical indicators
4. **HistoricalDataInitializer**: Fetches historical price data from DynamoDB to eliminate startup delay

## Commands

### Running Tests
```bash
# Basic integration tests (no TensorFlow required)
python3 tests/test_basic_integration.py

# Full ML integration tests (TensorFlow required)
python3 tests/test_ml_integration.py

# Historical data integration tests
python3 tests/test_historical_integration.py
```

### Installing Dependencies
```bash
# Basic dependencies
pip install -r requirements.txt

# For ML functionality
pip install tensorflow>=2.10.0
```

### Running the Trader
```bash
# With dummy model (default)
python3 trader.py

# With ML model (always enabled)
export ML_MODEL_PATH=models/best_model.weights.h5
python3 trader.py

# With historical data initialization (always enabled)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
python3 trader.py
```

## Environment Configuration

The system uses environment variables for configuration with sensible defaults:

### Trading Configuration
- `BITFLYER_API_KEY` / `BITFLYER_API_SECRET`: BitFlyer API credentials
- `ML_MODEL_PATH`: Path to trained model weights (default: `models/best_model.weights.h5`)

### Historical Data Configuration
- Historical data initialization is always enabled (no toggle required)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: AWS credentials
- `DYNAMODB_TABLE_NAME`: DynamoDB table (default: `crypto_currency_historicaldata`)
- `DYNAMODB_REGION`: AWS region (default: `ap-northeast-1`)
- `DYNAMODB_EXCHANGE_NAME`: Exchange identifier (default: `bitflyer-FX`)

## ML Model Integration

The system requires ML models to function:
- TensorFlow must be installed
- Model files must be available
- Falls back to standard initialization if DynamoDB is unavailable

### Model Requirements
- Trained weights from `liza_simple_predictor` project
- Input: 12-dimensional features (price, SMA, Bollinger, MACD, RSI)
- Output: 2-class probability (up/down direction)
- Input length (k): 90-360 minutes recommended
- Prediction horizon (p): typically 20-60 minutes

## Development Notes

### Error Handling Philosophy
The codebase uses extensive fallback mechanisms rather than hard failures. Comments throughout indicate "本来ならここに...エラーハンドリングが必要" (error handling needed here) showing areas where production error handling would be added.

### Critical File Restrictions
**⚠️ NEVER EDIT exchanges.py ⚠️**
- The `exchanges.py` file must NEVER be modified under any circumstances
- Error handling has been intentionally removed from this file during development
- This is a temporary development state and will be restored in production
- Any suggestions to improve exchanges.py should be noted but NOT implemented

### Code Structure Patterns
- Configuration through environment variables with defaults
- Modular design allowing independent testing of components
- Clear separation between API, models, and trading logic
- Extensive logging with prefixed messages for debugging

### Technical Indicators
The ML model uses sophisticated technical analysis:
- Multiple SMA windows (5, 20, 60 minutes)
- Bollinger Bands (20-minute window)
- MACD with signal line
- RSI (14-minute window)
- All indicators are normalized for ML input
- Variable naming convention: k (input length), p (prediction horizon), m (minutes)

### Testing Strategy
- Basic integration tests work without TensorFlow
- ML-specific tests require full TensorFlow installation
- Historical data tests validate DynamoDB integration
- Tests cover both success and fallback scenarios

## Code Style Guidelines

### Comment Style
All comments must be written in Japanese. Use the following format:

#### Module Documentation
```python
"""概要を簡潔に1行で記載する。

* 公開するクラス、関数などについて1行の説明を付けて一覧化する。(1行は72文字まで)
* ソースコードの始め(import文より前)に記載する。

"""
```

#### Function Documentation
```python
def func(arg1, arg2):
    """概要

    詳細説明

    Args:
        引数(arg1)の名前 (引数(arg1)の型): 引数(arg1)の説明
        引数(arg2)の名前 (:obj:`引数(arg2)の型`, optional): 引数(arg2)の説明

    Returns:
        戻り値の型: 戻り値の説明

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Examples:

        関数の使い方

        >>> func(5, 6)
        11

    Note:
        注意事項や注釈など

    """
    value = arg1 + arg2
    return value
```

#### Comment Requirements
- All docstrings and comments must be written in Japanese
- Use the Google-style docstring format with Japanese section headers
- Module docstrings should appear before import statements
- Keep single-line descriptions to 72 characters maximum
- Include type hints in both function signatures and docstrings