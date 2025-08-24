"""
Technical Indicators Module for NIFTY Options Backtesting System

This module provides a comprehensive set of technical indicators including:
MACD, RSI, ADX, SuperTrend, Bollinger Bands, EMA crossover, Stochastic, ATR

Author: Options Backtesting System
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Chainable technical indicators class for comprehensive market analysis.
    
    Usage:
        indicators = TechnicalIndicators(df)
        df_with_indicators = (indicators
                             .add_macd()
                             .add_rsi()
                             .add_bollinger_bands()
                             .add_supertrend()
                             .get_dataframe())
    """
    
    def __init__(self, df: pd.DataFrame, price_col: str = 'close'):
        """
        Initialize with OHLC data.
        
        Args:
            df: DataFrame with OHLC data
            price_col: Column name for price (default: 'close')
        """
        self.df = df.copy()
        self.price_col = price_col
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def add_sma(self, period: int = 20, column_name: Optional[str] = None) -> 'TechnicalIndicators':
        """Add Simple Moving Average."""
        col_name = column_name or f'sma_{period}'
        self.df[col_name] = self.df[self.price_col].rolling(window=period, min_periods=1).mean()
        return self
    
    def add_ema(self, period: int = 20, column_name: Optional[str] = None) -> 'TechnicalIndicators':
        """Add Exponential Moving Average."""
        col_name = column_name or f'ema_{period}'
        self.df[col_name] = self.df[self.price_col].ewm(span=period, adjust=False).mean()
        return self
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> 'TechnicalIndicators':
        """
        Add MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period  
            signal: Signal line EMA period
        """
        # Calculate MACD components
        ema_fast = self.df[self.price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df[self.price_col].ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # MACD buy/sell signals
        self.df['macd_bullish'] = (
            (self.df['macd'] > self.df['macd_signal']) & 
            (self.df['macd'].shift(1) <= self.df['macd_signal'].shift(1))
        )
        self.df['macd_bearish'] = (
            (self.df['macd'] < self.df['macd_signal']) & 
            (self.df['macd'].shift(1) >= self.df['macd_signal'].shift(1))
        )
        
        return self
    
    def add_rsi(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add RSI (Relative Strength Index) indicator.
        
        Args:
            period: RSI calculation period
        """
        delta = self.df[self.price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        self.df['rsi_oversold'] = self.df['rsi'] < 30
        self.df['rsi_overbought'] = self.df['rsi'] > 70
        
        return self
    
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> 'TechnicalIndicators':
        """
        Add Bollinger Bands indicator.
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
        """
        sma = self.df[self.price_col].rolling(window=period, min_periods=1).mean()
        std = self.df[self.price_col].rolling(window=period, min_periods=1).std()
        
        self.df['bb_upper'] = sma + (std * std_dev)
        self.df['bb_middle'] = sma
        self.df['bb_lower'] = sma - (std * std_dev)
        
        # Bollinger Bands signals
        self.df['bb_squeeze'] = (
            (self.df[self.price_col] <= self.df['bb_lower']) & 
            (self.df[self.price_col].shift(1) > self.df['bb_lower'].shift(1))
        )
        self.df['bb_breakout'] = (
            (self.df[self.price_col] >= self.df['bb_upper']) & 
            (self.df[self.price_col].shift(1) < self.df['bb_upper'].shift(1))
        )
        
        return self
    
    def add_atr(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add ATR (Average True Range) indicator.
        
        Args:
            period: ATR calculation period
        """
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift(1))
        low_close = np.abs(self.df['low'] - self.df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['atr'] = true_range.rolling(window=period, min_periods=1).mean()
        
        return self
    
    def add_supertrend(self, period: int = 10, multiplier: float = 3.0) -> 'TechnicalIndicators':
        """
        Add SuperTrend indicator.
        
        Args:
            period: ATR period
            multiplier: ATR multiplier
        """
        # First add ATR if not already present
        if 'atr' not in self.df.columns:
            self.add_atr(period)
        
        hl2 = (self.df['high'] + self.df['low']) / 2
        
        # Calculate basic upper and lower bands
        upper_band = hl2 + (multiplier * self.df['atr'])
        lower_band = hl2 - (multiplier * self.df['atr'])
        
        # Initialize SuperTrend
        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(index=self.df.index, dtype=int)
        
        for i in range(len(self.df)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                # Calculate final upper and lower bands
                if upper_band.iloc[i] < upper_band.iloc[i-1] or self.df['close'].iloc[i-1] > upper_band.iloc[i-1]:
                    final_upper_band = upper_band.iloc[i]
                else:
                    final_upper_band = upper_band.iloc[i-1]
                
                if lower_band.iloc[i] > lower_band.iloc[i-1] or self.df['close'].iloc[i-1] < lower_band.iloc[i-1]:
                    final_lower_band = lower_band.iloc[i]
                else:
                    final_lower_band = lower_band.iloc[i-1]
                
                # Determine SuperTrend direction
                if direction.iloc[i-1] == 1 and self.df['close'].iloc[i] <= final_lower_band:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = final_upper_band
                elif direction.iloc[i-1] == -1 and self.df['close'].iloc[i] >= final_upper_band:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = final_lower_band
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                    if direction.iloc[i] == 1:
                        supertrend.iloc[i] = final_lower_band
                    else:
                        supertrend.iloc[i] = final_upper_band
        
        self.df['supertrend'] = supertrend
        self.df['supertrend_direction'] = direction
        
        # SuperTrend signals
        self.df['supertrend_bullish'] = (
            (direction == 1) & (direction.shift(1) == -1)
        )
        self.df['supertrend_bearish'] = (
            (direction == -1) & (direction.shift(1) == 1)
        )
        
        return self
    
    def add_adx(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add ADX (Average Directional Index) indicator.
        
        Args:
            period: ADX calculation period
        """
        # Calculate True Range
        if 'atr' not in self.df.columns:
            self.add_atr(period)
        
        # Calculate Directional Movements
        plus_dm = self.df['high'].diff()
        minus_dm = -self.df['low'].diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate smoothed DM and TR
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / self.df['atr'])
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / self.df['atr'])
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=1).mean()
        
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        self.df['adx'] = adx
        
        # ADX signals (trend strength)
        self.df['adx_strong_trend'] = self.df['adx'] > 25
        self.df['adx_bullish'] = (self.df['plus_di'] > self.df['minus_di']) & self.df['adx_strong_trend']
        self.df['adx_bearish'] = (self.df['minus_di'] > self.df['plus_di']) & self.df['adx_strong_trend']
        
        return self
    
    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> 'TechnicalIndicators':
        """
        Add Stochastic Oscillator indicator.
        
        Args:
            k_period: %K period
            d_period: %D smoothing period
        """
        lowest_low = self.df['low'].rolling(window=k_period, min_periods=1).min()
        highest_high = self.df['high'].rolling(window=k_period, min_periods=1).max()
        
        k_percent = 100 * (self.df['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
        
        self.df['stoch_k'] = k_percent
        self.df['stoch_d'] = d_percent
        
        # Stochastic signals
        self.df['stoch_oversold'] = (self.df['stoch_k'] < 20) & (self.df['stoch_d'] < 20)
        self.df['stoch_overbought'] = (self.df['stoch_k'] > 80) & (self.df['stoch_d'] > 80)
        
        self.df['stoch_bullish'] = (
            (self.df['stoch_k'] > self.df['stoch_d']) & 
            (self.df['stoch_k'].shift(1) <= self.df['stoch_d'].shift(1)) & 
            self.df['stoch_oversold'].shift(1)
        )
        self.df['stoch_bearish'] = (
            (self.df['stoch_k'] < self.df['stoch_d']) & 
            (self.df['stoch_k'].shift(1) >= self.df['stoch_d'].shift(1)) & 
            self.df['stoch_overbought'].shift(1)
        )
        
        return self
    
    def add_ema_crossover(self, fast_period: int = 12, slow_period: int = 26) -> 'TechnicalIndicators':
        """
        Add EMA Crossover signals.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
        """
        fast_ema = self.df[self.price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.df[self.price_col].ewm(span=slow_period, adjust=False).mean()
        
        self.df[f'ema_{fast_period}'] = fast_ema
        self.df[f'ema_{slow_period}'] = slow_ema
        
        # EMA crossover signals
        self.df['ema_bullish_cross'] = (
            (fast_ema > slow_ema) & 
            (fast_ema.shift(1) <= slow_ema.shift(1))
        )
        self.df['ema_bearish_cross'] = (
            (fast_ema < slow_ema) & 
            (fast_ema.shift(1) >= slow_ema.shift(1))
        )
        
        return self
    
    def add_all_indicators(self) -> 'TechnicalIndicators':
        """Add all technical indicators with default parameters."""
        return (self
                .add_macd()
                .add_rsi()
                .add_bollinger_bands()
                .add_atr()
                .add_supertrend()
                .add_adx()
                .add_stochastic()
                .add_ema_crossover())
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the dataframe with all added indicators."""
        return self.df.copy()
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all current signals.
        
        Returns:
            Dictionary with signal counts and latest values
        """
        latest = self.df.iloc[-1]
        
        summary = {
            'latest_price': latest[self.price_col],
            'signals': {
                'macd_bullish': bool(latest.get('macd_bullish', False)),
                'macd_bearish': bool(latest.get('macd_bearish', False)),
                'rsi_oversold': bool(latest.get('rsi_oversold', False)),
                'rsi_overbought': bool(latest.get('rsi_overbought', False)),
                'bb_squeeze': bool(latest.get('bb_squeeze', False)),
                'bb_breakout': bool(latest.get('bb_breakout', False)),
                'supertrend_bullish': bool(latest.get('supertrend_bullish', False)),
                'supertrend_bearish': bool(latest.get('supertrend_bearish', False)),
                'adx_bullish': bool(latest.get('adx_bullish', False)),
                'adx_bearish': bool(latest.get('adx_bearish', False)),
                'stoch_bullish': bool(latest.get('stoch_bullish', False)),
                'stoch_bearish': bool(latest.get('stoch_bearish', False)),
                'ema_bullish_cross': bool(latest.get('ema_bullish_cross', False)),
                'ema_bearish_cross': bool(latest.get('ema_bearish_cross', False)),
            },
            'values': {
                'rsi': latest.get('rsi', np.nan),
                'adx': latest.get('adx', np.nan),
                'atr': latest.get('atr', np.nan),
                'stoch_k': latest.get('stoch_k', np.nan),
                'stoch_d': latest.get('stoch_d', np.nan),
            }
        }
        
        return summary


def main():
    """Example usage of TechnicalIndicators class."""
    # Sample data creation for testing
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLC data
    close_prices = 100 + np.random.randn(100).cumsum()
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    
    sample_df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    # Initialize indicators
    indicators = TechnicalIndicators(sample_df)
    
    # Add all indicators
    df_with_indicators = indicators.add_all_indicators().get_dataframe()
    
    print("Technical Indicators Added Successfully!")
    print(f"Original columns: {sample_df.columns.tolist()}")
    print(f"Final columns: {df_with_indicators.columns.tolist()}")
    print(f"\nDataFrame shape: {df_with_indicators.shape}")
    print(f"\nSignal Summary:")
    print(indicators.get_signal_summary())


if __name__ == "__main__":
    main()