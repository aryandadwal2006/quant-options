"""
Options Backtesting Engine for NIFTY Options Trading System

This module provides comprehensive backtesting capabilities including:
- Buy signal → Sell ATM PUT, Sell signal → Sell ATM CALL
- Risk management: 1.5% SL, 3% TP, 15:15 force exit
- Portfolio management and position tracking
- Performance metrics and reporting
- Equity curve and drawdown analysis

Author: Options Backtesting System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from utils import OptionsUtils


class Trade:
    """Individual trade representation."""
    
    def __init__(self, entry_time: pd.Timestamp, strike_price: float, 
                 option_type: str, entry_price: float, expiry_date: pd.Timestamp):
        self.entry_time = entry_time
        self.strike_price = strike_price
        self.option_type = option_type
        self.entry_price = entry_price
        self.expiry_date = expiry_date
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.is_open = True
        self.max_profit = 0.0
        self.max_loss = 0.0


class OptionsBacktester:
    """
    Comprehensive options backtesting engine.
    
    Features:
    - Strategy: Buy signal → Sell ATM PUT, Sell signal → Sell ATM CALL
    - Risk management with configurable parameters
    - Position tracking and portfolio management
    - Performance analytics and visualization
    - Detailed trade logging
    """
    
    def __init__(self, initial_capital: float = 200000, lot_size: int = 75,
                 stop_loss_pct: float = 1.5, take_profit_pct: float = 3.0,
                 force_exit_time: str = "15:15", timezone: str = 'Asia/Kolkata'):
        """
        Initialize backtester with configuration.
        
        Args:
            initial_capital: Starting capital in rupees
            lot_size: NIFTY lot size
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            force_exit_time: Daily force exit time (HH:MM)
            timezone: Timezone for operations
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.lot_size = lot_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.force_exit_time = force_exit_time
        self.timezone = timezone
        
        # Initialize utilities
        self.options_utils = OptionsUtils(lot_size, timezone)
        self.options_utils.stop_loss_pct = stop_loss_pct
        self.options_utils.take_profit_pct = take_profit_pct
        self.options_utils.force_exit_time = force_exit_time
        
        # Trading state
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.portfolio_history = []
        self.current_position = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
        
        # Data storage
        self.spot_data = None
        self.options_data = None
        self.signals_data = None
    
    def load_data(self, spot_df: pd.DataFrame, options_df: pd.DataFrame,
                  signals_df: Optional[pd.DataFrame] = None) -> None:
        """
        Load market data for backtesting.
        
        Args:
            spot_df: Spot price data with signals
            options_df: Options chain data
            signals_df: Additional signals data (optional)
        """
        self.spot_data = spot_df.copy()
        self.options_data = options_df.copy()
        
        # Ensure datetime columns are timezone-aware
        if 'datetime' in self.spot_data.columns:
            if self.spot_data['datetime'].dt.tz is None:
                self.spot_data['datetime'] = pd.to_datetime(self.spot_data['datetime']).dt.tz_localize(self.timezone)
        
        if 'datetime' in self.options_data.columns:
            if self.options_data['datetime'].dt.tz is None:
                self.options_data['datetime'] = pd.to_datetime(self.options_data['datetime']).dt.tz_localize(self.timezone)
        
        if 'expiry_date' in self.options_data.columns:
            self.options_data['expiry_date'] = pd.to_datetime(self.options_data['expiry_date'])
        
        # Merge signals if provided separately
        if signals_df is not None:
            self.signals_data = signals_df.copy()
            if 'datetime' in signals_df.columns:
                if self.signals_data['datetime'].dt.tz is None:
                    self.signals_data['datetime'] = pd.to_datetime(self.signals_data['datetime']).dt.tz_localize(self.timezone)
        
        print(f"Loaded spot data: {len(self.spot_data)} rows")
        print(f"Loaded options data: {len(self.options_data)} rows")
        print(f"Date range: {self.spot_data['datetime'].min()} to {self.spot_data['datetime'].max()}")
    
    def get_current_signal(self, timestamp: pd.Timestamp) -> str:
        """
        Get trading signal for given timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Signal: 'Buy', 'Sell', or 'Hold'
        """
        # Get signal from spot data
        spot_signal = self.spot_data[self.spot_data['datetime'] == timestamp]['composite_signal'].values
        
        if len(spot_signal) > 0:
            return spot_signal[0]
        
        # Fallback to original signal if composite not available
        orig_signal = self.spot_data[self.spot_data['datetime'] == timestamp]['signal'].values
        if len(orig_signal) > 0:
            signal = str(orig_signal[0]).strip().title()
            return signal if signal in ['Buy', 'Sell', 'Hold'] else 'Hold'
        
        return 'Hold'
    
    def get_spot_price(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get spot price for given timestamp."""
        spot_row = self.spot_data[self.spot_data['datetime'] == timestamp]
        if not spot_row.empty:
            return float(spot_row['close'].iloc[0])
        return None
    
    def find_atm_option(self, timestamp: pd.Timestamp, spot_price: float, 
                       option_type: str) -> Optional[Dict[str, Any]]:
        """
        Find ATM option for given parameters.
        
        Args:
            timestamp: Current timestamp
            spot_price: Current spot price
            option_type: 'CE' or 'PE'
            
        Returns:
            Dictionary with option details or None if not found
        """
        try:
            # Get available expiries for this timestamp
            available_expiries = self.options_data[
                (self.options_data['datetime'] == timestamp) &
                (self.options_data['option_type'] == option_type)
            ]['expiry_date'].unique()
            
            if len(available_expiries) == 0:
                return None
            
            # Find nearest expiry
            nearest_expiry = self.options_utils.find_nearest_expiry(
                timestamp, available_expiries
            )
            
            # Get available strikes for this expiry
            available_options = self.options_utils.get_available_options(
                self.options_data, timestamp, nearest_expiry, option_type
            )
            
            if available_options.empty:
                return None
            
            # Find ATM strike
            available_strikes = available_options['strike_price'].tolist()
            atm_strike = self.options_utils.find_atm_strike(spot_price, available_strikes)
            
            # Get option details
            atm_option = available_options[
                available_options['strike_price'] == atm_strike
            ].iloc[0]
            
            return {
                'strike_price': atm_strike,
                'option_type': option_type,
                'expiry_date': nearest_expiry,
                'entry_price': float(atm_option['close']),
                'entry_time': timestamp
            }
            
        except Exception as e:
            print(f"Error finding ATM option: {e}")
            return None
    
    def open_position(self, timestamp: pd.Timestamp, signal: str) -> bool:
        """
        Open new position based on signal.
        
        Args:
            timestamp: Entry timestamp
            signal: Trading signal ('Buy' or 'Sell')
            
        Returns:
            True if position opened successfully
        """
        try:
            spot_price = self.get_spot_price(timestamp)
            if spot_price is None:
                return False
            
            # Determine option type based on signal
            if signal == 'Buy':
                option_type = 'PE'  # Buy signal → Sell ATM PUT
            elif signal == 'Sell':
                option_type = 'CE'  # Sell signal → Sell ATM CALL
            else:
                return False
            
            # Find ATM option
            option_details = self.find_atm_option(timestamp, spot_price, option_type)
            if option_details is None:
                return False
            
            # Validate trade parameters
            validation = self.options_utils.validate_trade_params(
                spot_price, option_details['strike_price'],
                option_type, option_details['entry_price']
            )
            
            if not validation['is_valid']:
                print(f"Invalid trade parameters at {timestamp}: {validation['warnings']}")
                return False
            
            # Check if we have enough capital (simplified margin requirement)
            required_margin = option_details['entry_price'] * self.lot_size
            if self.current_capital < required_margin:
                print(f"Insufficient capital at {timestamp}: Need {required_margin}, Have {self.current_capital}")
                return False
            
            # Create trade
            trade = Trade(
                entry_time=timestamp,
                strike_price=option_details['strike_price'],
                option_type=option_type,
                entry_price=option_details['entry_price'],
                expiry_date=option_details['expiry_date']
            )
            
            # Add to open trades
            self.open_trades.append(trade)
            self.current_position = signal
            
            print(f"Opened {option_type} position at {timestamp}: Strike {trade.strike_price}, Premium {trade.entry_price}")
            return True
            
        except Exception as e:
            print(f"Error opening position: {e}")
            return False
    
    def check_exit_conditions(self, trade: Trade, timestamp: pd.Timestamp) -> Optional[Tuple[str, float]]:
        """
        Check if trade should be exited.
        
        Args:
            trade: Current trade
            timestamp: Current timestamp
            
        Returns:
            Tuple of (exit_reason, exit_price) or None if no exit
        """
        try:
            # Get current option price
            current_price = self.options_utils.get_options_price(
                self.options_data, timestamp, trade.strike_price,
                trade.option_type, trade.expiry_date, 'close'
            )
            
            if current_price is None:
                return None
            
            # Check force exit time
            if self.options_utils.should_force_exit(timestamp):
                return ('EOD', current_price)
            
            # Check stop loss
            if self.options_utils.check_stop_loss(trade.entry_price, current_price, 'short'):
                return ('Max SL', current_price)
            
            # Check take profit
            if self.options_utils.check_take_profit(trade.entry_price, current_price, 'short'):
                return ('Take Profit', current_price)
            
            # Check signal change
            current_signal = self.get_current_signal(timestamp)
            if current_signal != self.current_position and current_signal != 'Hold':
                return ('Signal Change', current_price)
            
            # Update max profit/loss tracking
            pnl = self.options_utils.calculate_option_pnl(
                trade.entry_price, current_price, 'short'
            )
            trade.max_profit = max(trade.max_profit, pnl)
            trade.max_loss = min(trade.max_loss, pnl)
            
            return None
            
        except Exception as e:
            print(f"Error checking exit conditions: {e}")
            return None
    
    def close_position(self, trade: Trade, timestamp: pd.Timestamp,
                      exit_reason: str, exit_price: float) -> None:
        """
        Close open position.
        
        Args:
            trade: Trade to close
            timestamp: Exit timestamp
            exit_reason: Reason for exit
            exit_price: Exit price
        """
        try:
            # Calculate P&L
            pnl = self.options_utils.calculate_option_pnl(
                trade.entry_price, exit_price, 'short'
            )
            
            # Update trade
            trade.exit_time = timestamp
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.pnl = pnl
            trade.is_open = False
            
            # Update portfolio
            self.current_capital += pnl
            self.total_pnl += pnl
            self.total_trades += 1
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update drawdown tracking
            self.peak_equity = max(self.peak_equity, self.current_capital)
            current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Move to completed trades
            self.trades.append(trade)
            self.open_trades.remove(trade)
            self.current_position = None
            
            print(f"Closed position at {timestamp}: P&L {pnl:.2f}, Reason: {exit_reason}")
            
        except Exception as e:
            print(f"Error closing position: {e}")
    
    def update_equity_curve(self, timestamp: pd.Timestamp) -> None:
        """Update equity curve with current portfolio value."""
        # Calculate unrealized P&L for open trades
        unrealized_pnl = 0.0
        for trade in self.open_trades:
            current_price = self.options_utils.get_options_price(
                self.options_data, timestamp, trade.strike_price,
                trade.option_type, trade.expiry_date, 'close'
            )
            if current_price is not None:
                unrealized_pnl += self.options_utils.calculate_option_pnl(
                    trade.entry_price, current_price, 'short'
                )
        
        # Current equity = cash + unrealized P&L
        current_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append({
            'datetime': timestamp,
            'equity': current_equity,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_pnl,
            'open_positions': len(self.open_trades)
        })
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run complete backtest.
        
        Returns:
            Dictionary with backtest results
        """
        if self.spot_data is None or self.options_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Starting backtest...")
        print(f"Initial Capital: ₹{self.initial_capital:,}")
        print(f"Stop Loss: {self.stop_loss_pct}%")
        print(f"Take Profit: {self.take_profit_pct}%")
        print(f"Force Exit Time: {self.force_exit_time}")
        
        # Get unique timestamps from spot data
        timestamps = sorted(self.spot_data['datetime'].unique())
        
        for i, timestamp in enumerate(timestamps):
            if i % 1000 == 0:
                print(f"Processing {i}/{len(timestamps)}: {timestamp}")
            
            # Update equity curve
            self.update_equity_curve(timestamp)
            
            # Check exit conditions for open trades
            trades_to_close = []
            for trade in self.open_trades:
                exit_condition = self.check_exit_conditions(trade, timestamp)
                if exit_condition:
                    exit_reason, exit_price = exit_condition
                    trades_to_close.append((trade, exit_reason, exit_price))
            
            # Close trades
            for trade, exit_reason, exit_price in trades_to_close:
                self.close_position(trade, timestamp, exit_reason, exit_price)
            
            # Check for new entries (only if no open positions)
            if len(self.open_trades) == 0:
                current_signal = self.get_current_signal(timestamp)
                if current_signal in ['Buy', 'Sell']:
                    self.open_position(timestamp, current_signal)
        
        # Close any remaining open positions at end of backtest
        for trade in self.open_trades.copy():
            final_timestamp = timestamps[-1]
            final_price = self.options_utils.get_options_price(
                self.options_data, final_timestamp, trade.strike_price,
                trade.option_type, trade.expiry_date, 'close'
            )
            if final_price is not None:
                self.close_position(trade, final_timestamp, 'End of Data', final_price)
        
        print(f"\nBacktest completed!")
        print(f"Total trades: {self.total_trades}")
        print(f"Winning trades: {self.winning_trades}")
        print(f"Losing trades: {self.losing_trades}")
        print(f"Final capital: ₹{self.current_capital:,.2f}")
        print(f"Total P&L: ₹{self.total_pnl:,.2f}")
        
        return self.generate_results()
    
    def generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if len(self.equity_curve) == 0:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate performance metrics
        total_return = (self.current_capital / self.initial_capital) - 1
        
        # Calculate Sharpe ratio (using daily returns)
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Win rate
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        # Average trade metrics
        trade_pnls = [trade.pnl for trade in self.trades]
        avg_trade = np.mean(trade_pnls) if trade_pnls else 0
        avg_winner = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0
        avg_loser = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if any(pnl < 0 for pnl in trade_pnls) else 0
        
        # Profit factor
        gross_profit = sum([pnl for pnl in trade_pnls if pnl > 0])
        gross_loss = abs(sum([pnl for pnl in trade_pnls if pnl < 0]))
        profit_factor = (gross_profit / max(gross_loss, 1)) if gross_loss > 0 else np.inf
        
        results = {
            'performance_metrics': {
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.max_drawdown * 100,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_trade': avg_trade,
                'avg_winner': avg_winner,
                'avg_loser': avg_loser,
                'final_capital': self.current_capital,
                'total_pnl': self.total_pnl
            },
            'equity_curve': equity_df,
            'trades': self.trades
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = 'results') -> None:
        """
        Save backtest results to files.
        
        Args:
            results: Results dictionary from run_backtest()
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([results['performance_metrics']])
        metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
        
        # Save trades
        if results['trades']:
            trades_data = []
            for trade in results['trades']:
                trades_data.append({
                    'strike_price': trade.strike_price,
                    'option_type': trade.option_type,
                    'entry_time': trade.entry_time,
                    'entry_price': trade.entry_price,
                    'exit_time': trade.exit_time,
                    'exit_price': trade.exit_price,
                    'exit_reason': trade.exit_reason,
                    'expiry_date': trade.expiry_date,
                    'm2m': 0,  # Placeholder for compatibility
                    'trade_date': trade.entry_time.date(),
                    'gross_pnl': trade.pnl,
                    'expenses': 0,  # Simplified
                    'interest': 0,  # Simplified  
                    'net_pnl': trade.pnl
                })
            
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(f'{output_dir}/trades.csv', index=False)
        
        # Save equity curve and generate plots
        equity_df = results['equity_curve']
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['datetime'], equity_df['equity'], linewidth=2, label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot drawdown
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_df['datetime'], equity_df['drawdown'], 0, 
                        alpha=0.3, color='red', label='Drawdown')
        plt.plot(equity_df['datetime'], equity_df['drawdown'], color='red', linewidth=1)
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to {output_dir}/ directory")


def main():
    """Example usage of OptionsBacktester."""
    # This would normally use real data
    # For demo purposes, we'll create minimal sample data
    
    print("Options Backtester Test")
    print("Note: This is a demonstration with minimal sample data")
    print("In practice, use real market data from your CSV files")
    
    # Create sample spot data
    dates = pd.date_range('2023-01-02 09:15', periods=100, freq='5T')
    np.random.seed(42)
    
    spot_data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.uniform(18000, 18200, 100),
        'high': np.random.uniform(18100, 18300, 100),
        'low': np.random.uniform(17900, 18100, 100),
        'close': np.random.uniform(18000, 18200, 100),
        'signal': np.random.choice(['Buy', 'Sell', 'Hold'], 100, p=[0.3, 0.3, 0.4]),
        'composite_signal': np.random.choice(['Buy', 'Sell', 'Hold'], 100, p=[0.25, 0.25, 0.5])
    })
    
    # Create sample options data
    options_data = []
    for date in dates[:10]:  # Limited for demo
        for strike in [18000, 18100, 18200]:
            for opt_type in ['CE', 'PE']:
                options_data.append({
                    'datetime': date,
                    'strike_price': strike,
                    'option_type': opt_type,
                    'expiry_date': pd.Timestamp('2023-01-26'),
                    'open': np.random.uniform(50, 150),
                    'high': np.random.uniform(100, 200),
                    'low': np.random.uniform(30, 100),
                    'close': np.random.uniform(60, 140),
                    'volume': np.random.randint(100, 1000),
                    'oi': np.random.randint(5000, 25000)
                })
    
    options_df = pd.DataFrame(options_data)
    
    # Initialize backtester
    backtester = OptionsBacktester(
        initial_capital=200000,
        stop_loss_pct=1.5,
        take_profit_pct=3.0,
        force_exit_time="15:15"
    )
    
    # Load data
    backtester.load_data(spot_data, options_df)
    
    print(f"\nData loaded successfully:")
    print(f"Spot data: {len(spot_data)} rows")
    print(f"Options data: {len(options_df)} rows")
    
    # Note: Full backtest would require complete options data
    print("\nBacktester initialized and ready for full market data!")
    print("To run complete backtest, load your actual CSV files and call:")
    print("results = backtester.run_backtest()")
    print("backtester.save_results(results)")


if __name__ == "__main__":
    main()