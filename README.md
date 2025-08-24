# NIFTY Options Backtesting System

A comprehensive, production-ready options trading backtesting system that combines technical indicators with machine learning models for systematic options strategies on NIFTY.

## 🎯 Overview

This system implements a sophisticated options trading strategy:
- **Buy Signal** → Sell ATM PUT option
- **Sell Signal** → Sell ATM CALL option
- **Risk Management**: 1.5% Stop Loss, 3% Take Profit, 15:15 Force Exit
- **Capital**: ₹200,000 starting capital
- **Technical Analysis**: 8+ indicators with weighted voting system
- **Machine Learning**: Multiple models for signal prediction
- **Comprehensive Reporting**: Equity curves, drawdowns, trade logs

## 📁 Project Structure

```
quant-options/
├── data/
│   ├── spot_with_signals_2023.csv      # Spot price data with original signals
│   └── options_data_2023.parquet       # Options chain data
├── indicators.py                       # Technical indicators (chainable)
├── signal_engine.py                    # Weighted voting signal system
├── model.py                           # ML models (Linear, XGBoost, LSTM)
├── backtest.py                        # Options backtesting engine
├── utils.py                           # Options utilities & risk management
├── main.py                            # Complete orchestration script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── results/                           # Generated outputs
    ├── equity_curve.png               # Portfolio performance chart
    ├── drawdown.png                   # Drawdown analysis
    ├── metrics.csv                    # Performance metrics
    ├── trades.csv                     # Individual trade records
    └── summary_report.csv             # Comprehensive summary
```

## 🛠️ Installation & Setup

### 1. Environment Setup

```bash
# Clone or download the project
cd quant-options

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Requirements

Your data files must match these formats exactly:

#### `spot_with_signals_2023.csv` Format:
```csv
datetime,open,high,low,close,closest_expiry,ap,esa,d,ci,tci,wt1,wt2,rsi,cross,signal
2023-01-02 09:20:00+05:30,18131.7,18150.15,18117.55,18141.35,2023-01-05,18125.43,18129.07,23.75,-10.22,-53.39,-53.39,-57.78,39.45,,Hold
```

#### `options_data_2023.parquet` Format:
```
underlying_symbol | ticker | datetime | expiry_date | strike_price | option_type | open | high | low | close | volume | oi
NIFTY | NIFTY29JUN2316000PE | 2023-01-02 09:15:00+05:30 | 2023-06-29 | 16000.0 | PE | 111.50 | 111.50 | 111.50 | 111.50 | 0 | 0
```

### 3. Required Dependencies

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
tensorflow>=2.8.0  # Optional for LSTM
matplotlib>=3.5.0
seaborn>=0.11.0
ta>=0.10.0  # Technical analysis library
pytz>=2022.1
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings
python main.py

# Specify custom data files
python main.py --spot-file data/your_spot_data.csv --options-file data/your_options_data.parquet

# Customize parameters
python main.py --capital 500000 --stop-loss 2.0 --take-profit 4.0

# Disable ML training for faster execution
python main.py --no-ml

# Custom output directory
python main.py --results-dir my_results
```

### Advanced Configuration

Create a `config.json` file for detailed customization:

```json
{
  "data": {
    "spot_file": "data/spot_with_signals_2023.csv",
    "options_file": "data/options_data_2023.parquet",
    "timezone": "Asia/Kolkata"
  },
  "indicators": {
    "macd_fast": 12,
    "macd_slow": 26,
    "rsi_period": 14,
    "bb_period": 20,
    "supertrend_period": 10
  },
  "signals": {
    "buy_threshold": 0.6,
    "sell_threshold": 0.6,
    "optimize_thresholds": true,
    "custom_weights": {
      "original_signal": 0.30,
      "macd_bullish": 0.15,
      "supertrend_bullish": 0.20,
      "rsi_oversold": 0.10
    }
  },
  "backtest": {
    "initial_capital": 200000,
    "lot_size": 75,
    "stop_loss_pct": 1.5,
    "take_profit_pct": 3.0,
    "force_exit_time": "15:15"
  },
  "ml": {
    "train_models": true,
    "models_to_train": ["linear", "xgboost", "lstm"]
  }
}
```

Run with custom config:
```bash
python main.py --config config.json
```

## 📊 Technical Indicators

The system includes 8 major technical indicators:

### 1. MACD (Moving Average Convergence Divergence)
- **Parameters**: Fast(12), Slow(26), Signal(9)
- **Signals**: Bullish/Bearish crossovers
- **Usage**: Trend following and momentum

### 2. RSI (Relative Strength Index)
- **Parameters**: Period(14)
- **Signals**: Oversold(<30), Overbought(>70)
- **Usage**: Momentum oscillator

### 3. Bollinger Bands
- **Parameters**: Period(20), StdDev(2.0)
- **Signals**: Squeeze, Breakout
- **Usage**: Volatility and mean reversion

### 4. SuperTrend
- **Parameters**: Period(10), Multiplier(3.0)
- **Signals**: Trend direction changes
- **Usage**: Strong trend following

### 5. ADX (Average Directional Index)
- **Parameters**: Period(14)
- **Signals**: Trend strength, Bullish/Bearish
- **Usage**: Trend strength measurement

### 6. Stochastic Oscillator
- **Parameters**: K(14), D(3)
- **Signals**: Oversold/Overbought crossovers
- **Usage**: Momentum oscillator

### 7. EMA Crossover
- **Parameters**: Fast(12), Slow(26)
- **Signals**: Golden/Death cross
- **Usage**: Trend identification

### 8. ATR (Average True Range)
- **Parameters**: Period(14)
- **Usage**: Volatility measurement, SuperTrend calculation

## 🧠 Machine Learning Models

### Supported Models
1. **Linear Regression** - For continuous predictions
2. **Logistic Regression** - For signal classification
3. **XGBoost** - Gradient boosting for both regression/classification
4. **Random Forest** - Ensemble method
5. **LSTM** - Deep learning for time series (requires TensorFlow)

### Feature Engineering
- Price changes (1, 2, 5 periods)
- Volatility measures (5, 10, 20 periods)
- Moving average ratios
- Technical indicator combinations
- Time-based features (hour, day of week)
- Lagged features

### Model Selection
- Time-series cross-validation
- Automatic best model selection
- Feature importance analysis
- Performance metrics comparison

## 📈 Options Strategy

### Core Strategy
- **Buy Signal** → Sell ATM PUT (profit if price goes up)
- **Sell Signal** → Sell ATM CALL (profit if price goes down)
- **Hold Signal** → No new positions

### Risk Management
- **Stop Loss**: 1.5% of premium (configurable)
- **Take Profit**: 3% of premium (configurable)
- **Force Exit**: 15:15 IST (end of day)
- **Position Sizing**: Based on available capital

### ATM Selection
- Find strike closest to current spot price
- Use nearest expiry available
- Validate option liquidity and pricing

## 📋 Signal Engine

### Weighted Voting System
Combines multiple signals with configurable weights:

```python
default_weights = {
    'original_signal': 0.25,      # Original signal from data
    'macd_bullish': 0.12,         # MACD buy signal
    'macd_bearish': 0.12,         # MACD sell signal
    'supertrend_bullish': 0.15,   # SuperTrend buy
    'supertrend_bearish': 0.15,   # SuperTrend sell
    'rsi_oversold': 0.08,         # RSI oversold
    'rsi_overbought': 0.08,       # RSI overbought
    # ... other indicators
}
```

### Composite Signal Generation
- **Buy**: Total buy score ≥ buy_threshold (default: 0.6)
- **Sell**: Total sell score ≥ sell_threshold (default: 0.6)
- **Hold**: Neither threshold met

### Threshold Optimization
- Automatic optimization based on historical returns
- Sharpe ratio maximization
- Time-series validation

## 📊 Output Files

### 1. `metrics.csv`
Key performance metrics:
```csv
total_return,sharpe_ratio,max_drawdown,win_rate,profit_factor,total_trades
15.25,1.234,8.75,65.5,1.85,245
```

### 2. `trades.csv`
Individual trade records:
```csv
strike_price,option_type,entry_time,entry_price,exit_time,exit_price,exit_reason,pnl
18000,PE,2023-01-02 09:30:00,95.5,2023-01-02 11:15:00,75.2,Take Profit,1522.5
```

### 3. `equity_curve.png`
Portfolio value over time with:
- Daily equity curve
- Initial capital baseline
- Drawdown periods highlighted

### 4. `drawdown.png`
Underwater equity curve showing:
- Percentage drawdowns
- Maximum drawdown periods
- Recovery periods

### 5. `summary_report.csv`
Comprehensive summary with:
- Configuration parameters
- Data statistics
- Signal performance
- Backtest results

## ⚙️ Configuration Options

### Data Configuration
```python
'data': {
    'spot_file': 'path/to/spot_data.csv',
    'options_file': 'path/to/options_data.parquet',
    'timezone': 'Asia/Kolkata'
}
```

### Indicator Parameters
```python
'indicators': {
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std': 2.0,
    'supertrend_period': 10,
    'supertrend_multiplier': 3.0,
    'adx_period': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'ema_fast': 12,
    'ema_slow': 26
}
```

### Signal Weights
```python
'signals': {
    'buy_threshold': 0.6,
    'sell_threshold': 0.6,
    'optimize_thresholds': True,
    'custom_weights': {
        'original_signal': 0.25,
        'macd_bullish': 0.12,
        'supertrend_bullish': 0.15,
        # ... customize as needed
    }
}
```

### Backtest Parameters
```python
'backtest': {
    'initial_capital': 200000,
    'lot_size': 75,
    'stop_loss_pct': 1.5,
    'take_profit_pct': 3.0,
    'force_exit_time': '15:15'
}
```

## 🔧 Advanced Usage

### Custom Indicator Weights
```python
from main import NiftyOptionsBacktester

# Custom configuration
config = {
    'signals': {
        'custom_weights': {
            'original_signal': 0.30,
            'supertrend_bullish': 0.25,
            'supertrend_bearish': 0.25,
            'macd_bullish': 0.10,
            'macd_bearish': 0.10
        }
    }
}

# Run backtester
backtester = NiftyOptionsBacktester(config)
results = backtester.run_complete_pipeline()
```

### ML Model Selection
```python
config = {
    'ml': {
        'train_models': True,
        'models_to_train': ['xgboost', 'random_forest'],  # Only specific models
        'target_column': 'composite_signal'
    }
}
```

### Risk Management Customization
```python
config = {
    'backtest': {
        'stop_loss_pct': 2.0,      # 2% stop loss
        'take_profit_pct': 5.0,    # 5% take profit
        'force_exit_time': '15:00'  # Earlier exit
    }
}
```

## 📈 Performance Interpretation

### Key Metrics

#### Total Return
- Percentage return on initial capital
- **Good**: >15% annually
- **Excellent**: >25% annually

#### Sharpe Ratio
- Risk-adjusted returns
- **Good**: >1.0
- **Excellent**: >2.0

#### Maximum Drawdown
- Largest peak-to-trough decline
- **Good**: <15%
- **Excellent**: <10%

#### Win Rate
- Percentage of profitable trades
- **Good**: >55%
- **Excellent**: >65%

#### Profit Factor
- Ratio of gross profits to gross losses
- **Good**: >1.5
- **Excellent**: >2.0

### Signal Quality Indicators
- **Signal Distribution**: Balanced Buy/Sell/Hold signals
- **Signal Strength**: Higher average strength for profitable signals
- **Signal Transitions**: Reasonable frequency of signal changes

## 🐛 Troubleshooting

### Common Issues

#### 1. Data Loading Errors
```
Error: Missing required columns
```
**Solution**: Ensure your CSV files match the exact format specified

#### 2. Timezone Issues
```
TypeError: Cannot localize tz-aware datetime
```
**Solution**: Check timezone settings in config, ensure consistent timezone handling

#### 3. Memory Issues with Large Datasets
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Reduce data size or timeframe
- Disable ML models with `--no-ml`
- Use more powerful hardware

#### 4. Options Data Not Found
```
No options data found for timestamp
```
**Solution**: 
- Verify options data covers the same time range as spot data
- Check timestamp alignment
- Ensure adequate options chain coverage

#### 5. TensorFlow/LSTM Issues
```
TensorFlow not available. LSTM models will not be supported.
```
**Solution**: Install TensorFlow: `pip install tensorflow`

### Performance Issues

#### 1. Slow Execution
- Disable ML training: `--no-ml`
- Reduce data size for testing
- Use SSD storage for data files

#### 2. Poor Backtest Results
- Adjust signal thresholds
- Optimize indicator parameters
- Review signal weights distribution
- Check data quality and coverage

## 📚 Code Architecture

### Modular Design
- **indicators.py**: Chainable technical indicators
- **signal_engine.py**: Weighted signal combination
- **model.py**: ML model training and prediction
- **utils.py**: Options utilities and risk management
- **backtest.py**: Core backtesting engine
- **main.py**: Pipeline orchestration

### Key Classes
- `TechnicalIndicators`: Chainable indicator calculations
- `SignalEngine`: Weighted voting signal system
- `MLPredictor`: Machine learning model manager
- `OptionsUtils`: Options-specific utilities
- `OptionsBacktester`: Complete backtesting engine
- `NiftyOptionsBacktester`: Main orchestrator

### Error Handling
- Comprehensive try-catch blocks
- Graceful degradation for optional components
- Detailed error messages and logging
- Data validation at each stage

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/aryandadwal2006/quant-options
cd strategy-backtest

# Install development dependencies
pip install -r requirements.txt

```

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints for function parameters
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only. 

- **Not Financial Advice**: This system does not provide financial advice
- **No Guarantees**: Past performance does not guarantee future results
- **Risk Warning**: Options trading involves substantial risk
- **Use at Your Own Risk**: Users are responsible for their trading decisions
- **Validate Results**: Always validate backtest results with out-of-sample data

## 📞 Support

For questions, issues, or feature requests:

1. **Documentation**: Check this README and code comments
2. **Issues**: Create a GitHub issue with detailed description
3. **Discussions**: Use GitHub discussions for questions
4. **Email**: Contact the development team

## 🔄 Version History

### v1.0.0 (Current)
- Initial release
- Complete options backtesting system
- 8 technical indicators
- 5 ML models
- Comprehensive reporting
- Configuration-driven architecture

### Upcoming Features
- Real-time trading integration
- More advanced ML models
- Portfolio optimization
- Risk analytics dashboard
- Web-based interface

---


**Happy Trading! 📈**
