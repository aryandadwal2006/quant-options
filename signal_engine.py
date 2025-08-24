"""
Signal Engine Module for NIFTY Options Backtesting System

This module provides a weighted voting system to combine original signals 
with technical indicators to generate composite Buy/Sell/Hold signals.

Author: Options Backtesting System
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class SignalEngine:
    """
    Weighted voting system for combining multiple trading signals.
    
    Features:
    - Configurable weights for each signal type
    - Support for original signals and technical indicators
    - Composite signal generation with thresholds
    - Signal strength analysis
    - Historical signal performance tracking
    """
    
    def __init__(self, 
                 buy_threshold: float = 0.6, 
                 sell_threshold: float = 0.6,
                 signal_weights: Optional[Dict[str, float]] = None):
        """
        Initialize SignalEngine with thresholds and weights.
        
        Args:
            buy_threshold: Threshold for generating BUY signals (0-1)
            sell_threshold: Threshold for generating SELL signals (0-1)  
            signal_weights: Dictionary of signal weights
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # Default signal weights - can be customized
        self.default_weights = {
            # Original signal from data
            'original_signal': 0.25,
            
            # MACD signals
            'macd_bullish': 0.12,
            'macd_bearish': 0.12,
            
            # RSI signals
            'rsi_oversold': 0.08,
            'rsi_overbought': 0.08,
            
            # Bollinger Bands signals
            'bb_squeeze': 0.08,
            'bb_breakout': 0.08,
            
            # SuperTrend signals
            'supertrend_bullish': 0.15,
            'supertrend_bearish': 0.15,
            
            # ADX signals
            'adx_bullish': 0.10,
            'adx_bearish': 0.10,
            
            # Stochastic signals
            'stoch_bullish': 0.07,
            'stoch_bearish': 0.07,
            
            # EMA Crossover signals
            'ema_bullish_cross': 0.08,
            'ema_bearish_cross': 0.08,
        }
        
        # Use provided weights or defaults
        self.signal_weights = signal_weights or self.default_weights.copy()
        
        # Validate weights sum approximately to 1.0
        total_weight = sum(self.signal_weights.values())
        if abs(total_weight - 1.0) > 0.1:  # Allow 10% tolerance
            print(f"Warning: Signal weights sum to {total_weight:.3f}, not 1.0")
    
    def normalize_original_signal(self, signal: str) -> Tuple[float, float]:
        """
        Convert original signal strings to buy/sell scores.
        
        Args:
            signal: Original signal ('Buy', 'Sell', 'Hold', etc.)
            
        Returns:
            Tuple of (buy_score, sell_score)
        """
        signal = str(signal).strip().lower()
        
        if signal in ['buy', 'bullish', '1', 'long']:
            return (1.0, 0.0)
        elif signal in ['sell', 'bearish', '-1', 'short']:
            return (0.0, 1.0)
        elif signal in ['hold', 'neutral', '0']:
            return (0.0, 0.0)
        else:
            # Unknown signal - treat as hold
            return (0.0, 0.0)
    
    def calculate_signal_scores(self, row: pd.Series) -> Tuple[float, float]:
        """
        Calculate weighted buy and sell scores for a single row.
        
        Args:
            row: DataFrame row with signal columns
            
        Returns:
            Tuple of (buy_score, sell_score)
        """
        buy_score = 0.0
        sell_score = 0.0
        
        # Process original signal if present
        if 'signal' in row and 'original_signal' in self.signal_weights:
            orig_buy, orig_sell = self.normalize_original_signal(row['signal'])
            buy_score += orig_buy * self.signal_weights['original_signal']
            sell_score += orig_sell * self.signal_weights['original_signal']
        
        # Process technical indicator signals
        signal_mappings = {
            # Bullish signals
            'macd_bullish': ('buy', 0.0),
            'rsi_oversold': ('buy', 0.0),
            'bb_squeeze': ('buy', 0.0),
            'supertrend_bullish': ('buy', 0.0),
            'adx_bullish': ('buy', 0.0),
            'stoch_bullish': ('buy', 0.0),
            'ema_bullish_cross': ('buy', 0.0),
            
            # Bearish signals
            'macd_bearish': ('sell', 0.0),
            'rsi_overbought': ('sell', 0.0),
            'bb_breakout': ('sell', 0.0),
            'supertrend_bearish': ('sell', 0.0),
            'adx_bearish': ('sell', 0.0),
            'stoch_bearish': ('sell', 0.0),
            'ema_bearish_cross': ('sell', 0.0),
        }
        
        for signal_name, (signal_type, _) in signal_mappings.items():
            if signal_name in row and signal_name in self.signal_weights:
                signal_value = row[signal_name]
                weight = self.signal_weights[signal_name]
                
                # Convert boolean or numeric signals to score
                if isinstance(signal_value, (bool, np.bool_)):
                    signal_strength = 1.0 if signal_value else 0.0
                else:
                    # Handle numeric signals (assume 0-1 range or boolean-like)
                    signal_strength = float(signal_value) if not pd.isna(signal_value) else 0.0
                
                if signal_type == 'buy':
                    buy_score += signal_strength * weight
                else:  # signal_type == 'sell'
                    sell_score += signal_strength * weight
        
        return buy_score, sell_score
    
    def generate_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite signals for the entire dataframe.
        
        Args:
            df: DataFrame with signal columns
            
        Returns:
            DataFrame with added composite signal columns
        """
        result_df = df.copy()
        
        # Initialize score columns
        buy_scores = []
        sell_scores = []
        composite_signals = []
        signal_strengths = []
        
        for idx, row in df.iterrows():
            buy_score, sell_score = self.calculate_signal_scores(row)
            
            # Determine composite signal
            if buy_score >= self.buy_threshold and buy_score > sell_score:
                composite_signal = 'Buy'
                signal_strength = buy_score
            elif sell_score >= self.sell_threshold and sell_score > buy_score:
                composite_signal = 'Sell'
                signal_strength = sell_score
            else:
                composite_signal = 'Hold'
                signal_strength = max(buy_score, sell_score)
            
            buy_scores.append(buy_score)
            sell_scores.append(sell_score)
            composite_signals.append(composite_signal)
            signal_strengths.append(signal_strength)
        
        # Add columns to dataframe
        result_df['buy_score'] = buy_scores
        result_df['sell_score'] = sell_scores
        result_df['composite_signal'] = composite_signals
        result_df['signal_strength'] = signal_strengths
        
        # Add signal change detection
        result_df['signal_changed'] = (
            result_df['composite_signal'] != result_df['composite_signal'].shift(1)
        )
        
        return result_df
    
    def optimize_thresholds(self, df: pd.DataFrame, 
                          price_col: str = 'close',
                          lookforward_periods: int = 5) -> Dict[str, float]:
        """
        Optimize buy/sell thresholds based on forward returns.
        
        Args:
            df: DataFrame with signals and prices
            price_col: Price column name
            lookforward_periods: Periods to look forward for returns
            
        Returns:
            Dictionary with optimal thresholds
        """
        if len(df) < lookforward_periods + 10:
            return {'buy_threshold': self.buy_threshold, 'sell_threshold': self.sell_threshold}
        
        # Calculate forward returns
        forward_returns = df[price_col].shift(-lookforward_periods) / df[price_col] - 1
        
        best_sharpe = -np.inf
        best_thresholds = {'buy_threshold': self.buy_threshold, 'sell_threshold': self.sell_threshold}
        
        # Test different threshold combinations
        for buy_thresh in np.arange(0.3, 1.0, 0.1):
            for sell_thresh in np.arange(0.3, 1.0, 0.1):
                # Temporarily update thresholds
                old_buy, old_sell = self.buy_threshold, self.sell_threshold
                self.buy_threshold, self.sell_threshold = buy_thresh, sell_thresh
                
                # Generate signals with new thresholds
                temp_df = self.generate_composite_signals(df)
                
                # Calculate returns for each signal type
                buy_returns = forward_returns[temp_df['composite_signal'] == 'Buy']
                sell_returns = -forward_returns[temp_df['composite_signal'] == 'Sell']  # Short returns
                
                if len(buy_returns) > 0 or len(sell_returns) > 0:
                    all_returns = pd.concat([buy_returns, sell_returns])
                    if len(all_returns) > 0 and all_returns.std() > 0:
                        sharpe = all_returns.mean() / all_returns.std()
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresholds = {'buy_threshold': buy_thresh, 'sell_threshold': sell_thresh}
                
                # Restore original thresholds
                self.buy_threshold, self.sell_threshold = old_buy, old_sell
        
        return best_thresholds
    
    def get_signal_performance(self, df: pd.DataFrame, 
                             price_col: str = 'close') -> Dict[str, Any]:
        """
        Analyze signal performance and statistics.
        
        Args:
            df: DataFrame with signals and prices
            price_col: Price column name
            
        Returns:
            Dictionary with performance metrics
        """
        if 'composite_signal' not in df.columns:
            df = self.generate_composite_signals(df)
        
        # Calculate basic statistics
        signal_counts = df['composite_signal'].value_counts()
        total_signals = len(df)
        
        # Calculate signal transitions
        transitions = df[df['signal_changed'] == True]['composite_signal'].value_counts()
        
        # Calculate average signal strength by type
        avg_strength = df.groupby('composite_signal')['signal_strength'].mean()
        
        performance = {
            'signal_distribution': {
                'buy_count': int(signal_counts.get('Buy', 0)),
                'sell_count': int(signal_counts.get('Sell', 0)),
                'hold_count': int(signal_counts.get('Hold', 0)),
                'buy_percentage': float(signal_counts.get('Buy', 0) / total_signals * 100),
                'sell_percentage': float(signal_counts.get('Sell', 0) / total_signals * 100),
                'hold_percentage': float(signal_counts.get('Hold', 0) / total_signals * 100),
            },
            'signal_transitions': {
                'buy_transitions': int(transitions.get('Buy', 0)),
                'sell_transitions': int(transitions.get('Sell', 0)),
                'hold_transitions': int(transitions.get('Hold', 0)),
            },
            'average_signal_strength': {
                'buy_strength': float(avg_strength.get('Buy', 0)),
                'sell_strength': float(avg_strength.get('Sell', 0)),
                'hold_strength': float(avg_strength.get('Hold', 0)),
            },
            'configuration': {
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'total_weight_sum': sum(self.signal_weights.values()),
            }
        }
        
        return performance
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update signal weights.
        
        Args:
            new_weights: Dictionary of new weights
        """
        self.signal_weights.update(new_weights)
        
        # Warn if weights don't sum to ~1.0
        total_weight = sum(self.signal_weights.values())
        if abs(total_weight - 1.0) > 0.1:
            print(f"Warning: Updated weights sum to {total_weight:.3f}, not 1.0")
    
    def get_active_signals(self, df: pd.DataFrame, 
                          latest_only: bool = True) -> Dict[str, Any]:
        """
        Get currently active signals.
        
        Args:
            df: DataFrame with signals
            latest_only: If True, return only latest row signals
            
        Returns:
            Dictionary with active signals
        """
        if latest_only:
            if len(df) == 0:
                return {}
            
            latest_row = df.iloc[-1]
            buy_score, sell_score = self.calculate_signal_scores(latest_row)
            
            # Get individual signal contributions
            active_signals = {}
            
            # Check original signal
            if 'signal' in latest_row and 'original_signal' in self.signal_weights:
                orig_buy, orig_sell = self.normalize_original_signal(latest_row['signal'])
                if orig_buy > 0 or orig_sell > 0:
                    active_signals['original_signal'] = {
                        'value': str(latest_row['signal']),
                        'buy_contribution': orig_buy * self.signal_weights['original_signal'],
                        'sell_contribution': orig_sell * self.signal_weights['original_signal']
                    }
            
            # Check technical signals
            for signal_name, weight in self.signal_weights.items():
                if signal_name != 'original_signal' and signal_name in latest_row:
                    signal_value = latest_row[signal_name]
                    if isinstance(signal_value, (bool, np.bool_)) and signal_value:
                        is_buy_signal = signal_name in [
                            'macd_bullish', 'rsi_oversold', 'bb_squeeze', 
                            'supertrend_bullish', 'adx_bullish', 'stoch_bullish', 'ema_bullish_cross'
                        ]
                        active_signals[signal_name] = {
                            'value': True,
                            'buy_contribution': weight if is_buy_signal else 0.0,
                            'sell_contribution': weight if not is_buy_signal else 0.0
                        }
            
            return {
                'timestamp': latest_row.get('datetime', 'Unknown'),
                'total_buy_score': buy_score,
                'total_sell_score': sell_score,
                'composite_signal': 'Buy' if buy_score >= self.buy_threshold and buy_score > sell_score 
                                   else 'Sell' if sell_score >= self.sell_threshold and sell_score > buy_score 
                                   else 'Hold',
                'active_signals': active_signals,
                'signal_count': len(active_signals)
            }
        else:
            # Return all active signals (implementation for historical analysis)
            return self.get_signal_performance(df)
    
    def backtest_signals(self, df: pd.DataFrame, 
                        price_col: str = 'close',
                        lookforward_periods: int = 5) -> pd.DataFrame:
        """
        Backtest signal performance.
        
        Args:
            df: DataFrame with signals and prices
            price_col: Price column name
            lookforward_periods: Periods to hold position
            
        Returns:
            DataFrame with backtest results
        """
        if 'composite_signal' not in df.columns:
            df = self.generate_composite_signals(df)
        
        results_df = df.copy()
        
        # Calculate forward returns
        forward_returns = df[price_col].shift(-lookforward_periods) / df[price_col] - 1
        
        # Calculate strategy returns based on signals
        strategy_returns = []
        
        for idx, row in df.iterrows():
            if row['composite_signal'] == 'Buy':
                # Long position return
                ret = forward_returns.iloc[idx] if idx < len(forward_returns) else 0
            elif row['composite_signal'] == 'Sell':
                # Short position return (negative of forward return)
                ret = -forward_returns.iloc[idx] if idx < len(forward_returns) else 0
            else:  # Hold
                ret = 0
            
            strategy_returns.append(ret if not pd.isna(ret) else 0)
        
        results_df['forward_return'] = forward_returns
        results_df['strategy_return'] = strategy_returns
        results_df['cumulative_return'] = (1 + pd.Series(strategy_returns)).cumprod() - 1
        
        return results_df


def main():
    """Example usage of SignalEngine class."""
    # Create sample data with signals
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLC data
    close_prices = 100 + np.random.randn(100).cumsum()
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    
    # Generate sample signals
    original_signals = np.random.choice(['Buy', 'Sell', 'Hold'], 100, p=[0.3, 0.3, 0.4])
    
    sample_df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'signal': original_signals,
        # Add some sample technical indicator signals
        'macd_bullish': np.random.random(100) > 0.8,
        'macd_bearish': np.random.random(100) > 0.8,
        'rsi_oversold': np.random.random(100) > 0.9,
        'rsi_overbought': np.random.random(100) > 0.9,
        'supertrend_bullish': np.random.random(100) > 0.85,
        'supertrend_bearish': np.random.random(100) > 0.85,
    })
    
    # Initialize signal engine
    signal_engine = SignalEngine(
        buy_threshold=0.6,
        sell_threshold=0.6
    )
    
    # Generate composite signals
    df_with_signals = signal_engine.generate_composite_signals(sample_df)
    
    print("Signal Engine Test Results:")
    print(f"Original DataFrame shape: {sample_df.shape}")
    print(f"DataFrame with composite signals shape: {df_with_signals.shape}")
    
    # Show signal performance
    performance = signal_engine.get_signal_performance(df_with_signals)
    print(f"\nSignal Performance:")
    print(f"Buy signals: {performance['signal_distribution']['buy_count']} ({performance['signal_distribution']['buy_percentage']:.1f}%)")
    print(f"Sell signals: {performance['signal_distribution']['sell_count']} ({performance['signal_distribution']['sell_percentage']:.1f}%)")
    print(f"Hold signals: {performance['signal_distribution']['hold_count']} ({performance['signal_distribution']['hold_percentage']:.1f}%)")
    
    # Show latest active signals
    active_signals = signal_engine.get_active_signals(df_with_signals)
    print(f"\nLatest Active Signals:")
    print(f"Composite Signal: {active_signals['composite_signal']}")
    print(f"Buy Score: {active_signals['total_buy_score']:.3f}")
    print(f"Sell Score: {active_signals['total_sell_score']:.3f}")
    print(f"Active Signal Count: {active_signals['signal_count']}")
    
    # Test backtest functionality
    backtest_results = signal_engine.backtest_signals(df_with_signals)
    final_return = backtest_results['cumulative_return'].iloc[-1]
    print(f"\nBacktest Results:")
    print(f"Final Cumulative Return: {final_return:.2%}")
    
    # Test threshold optimization
    optimal_thresholds = signal_engine.optimize_thresholds(df_with_signals)
    print(f"\nOptimal Thresholds:")
    print(f"Buy Threshold: {optimal_thresholds['buy_threshold']:.2f}")
    print(f"Sell Threshold: {optimal_thresholds['sell_threshold']:.2f}")


if __name__ == "__main__":
    main()