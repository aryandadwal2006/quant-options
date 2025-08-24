"""
Options Trading Utilities Module for NIFTY Options Backtesting System

This module provides utility functions for:
- ATM strike selection
- Nearest expiry finding  
- Options price retrieval
- P&L calculations for short positions
- Stop loss/take profit checking
- Risk management

Author: Options Backtesting System
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class OptionsUtils:
    """
    Comprehensive utilities for options trading operations.
    
    Features:
    - ATM strike selection
    - Options data filtering and retrieval
    - P&L calculations for short options positions
    - Risk management (stop loss, take profit)
    - Expiry management
    - Trade validation
    """
    
    def __init__(self, lot_size: int = 75, timezone: str = 'Asia/Kolkata'):
        """
        Initialize OptionsUtils.
        
        Args:
            lot_size: NIFTY lot size (default: 75)
            timezone: Timezone for datetime operations
        """
        self.lot_size = lot_size
        self.timezone = timezone
        
        # Risk management parameters
        self.stop_loss_pct = 1.5  # 1.5% stop loss
        self.take_profit_pct = 3.0  # 3% take profit
        self.force_exit_time = "15:15"  # Force exit time
    
    def find_atm_strike(self, spot_price: float, available_strikes: List[float]) -> float:
        """
        Find At-The-Money (ATM) strike closest to spot price.
        
        Args:
            spot_price: Current spot price
            available_strikes: List of available strike prices
            
        Returns:
            ATM strike price
        """
        if not available_strikes:
            raise ValueError("No available strikes provided")
        
        available_strikes = sorted(available_strikes)
        
        # Find closest strike to spot price
        differences = [abs(strike - spot_price) for strike in available_strikes]
        min_diff_idx = differences.index(min(differences))
        
        return available_strikes[min_diff_idx]
    
    def find_nearest_expiry(self, current_datetime: pd.Timestamp, 
                          available_expiries: List[pd.Timestamp]) -> pd.Timestamp:
        """
        Find the nearest expiry date from available options.
        
        Args:
            current_datetime: Current timestamp
            available_expiries: List of available expiry dates
            
        Returns:
            Nearest expiry timestamp
        """
        if not available_expiries:
            raise ValueError("No available expiries provided")
        
        # Convert to pandas timestamps if needed
        available_expiries = [pd.Timestamp(exp) for exp in available_expiries]
        
        # Filter future expiries only
        future_expiries = [exp for exp in available_expiries if exp > current_datetime]
        
        if not future_expiries:
            # If no future expiries, return the latest available
            return max(available_expiries)
        
        # Find nearest future expiry
        time_differences = [(exp - current_datetime).total_seconds() for exp in future_expiries]
        min_diff_idx = time_differences.index(min(time_differences))
        
        return future_expiries[min_diff_idx]
    
    def get_options_price(self, options_df: pd.DataFrame, 
                         timestamp: pd.Timestamp,
                         strike: float,
                         option_type: str,
                         expiry_date: pd.Timestamp,
                         price_type: str = 'close') -> Optional[float]:
        """
        Get options price for specific parameters.
        
        Args:
            options_df: Options data DataFrame
            timestamp: Trade timestamp
            strike: Strike price
            option_type: 'CE' or 'PE'
            expiry_date: Option expiry date
            price_type: Price type ('open', 'high', 'low', 'close')
            
        Returns:
            Options price or None if not found
        """
        try:
            # Filter options data
            filtered_data = options_df[
                (options_df['datetime'] == timestamp) &
                (options_df['strike_price'] == strike) &
                (options_df['option_type'] == option_type) &
                (options_df['expiry_date'] == expiry_date)
            ]
            
            if filtered_data.empty:
                # Try to find nearest timestamp within same minute
                timestamp_minute = timestamp.floor('min')
                time_window = options_df[
                    (options_df['datetime'].dt.floor('min') == timestamp_minute) &
                    (options_df['strike_price'] == strike) &
                    (options_df['option_type'] == option_type) &
                    (options_df['expiry_date'] == expiry_date)
                ]
                
                if time_window.empty:
                    return None
                
                # Get closest timestamp
                filtered_data = time_window.iloc[[0]]
            
            price = filtered_data[price_type].iloc[0]
            return float(price) if not pd.isna(price) else None
            
        except Exception as e:
            print(f"Error getting options price: {e}")
            return None
    
    def calculate_option_pnl(self, entry_price: float, 
                           exit_price: float,
                           position_type: str = 'short') -> float:
        """
        Calculate P&L for options position.
        
        Args:
            entry_price: Entry premium
            exit_price: Exit premium
            position_type: 'short' (selling) or 'long' (buying)
            
        Returns:
            P&L amount in rupees
        """
        if position_type.lower() == 'short':
            # For short positions: profit when option price decreases
            raw_pnl = (entry_price - exit_price) * self.lot_size
        else:
            # For long positions: profit when option price increases
            raw_pnl = (exit_price - entry_price) * self.lot_size
        
        return raw_pnl
    
    def check_stop_loss(self, entry_price: float, 
                       current_price: float,
                       position_type: str = 'short') -> bool:
        """
        Check if stop loss is hit.
        
        Args:
            entry_price: Entry premium
            current_price: Current premium
            position_type: 'short' or 'long'
            
        Returns:
            True if stop loss is hit
        """
        if position_type.lower() == 'short':
            # For short positions, stop loss when price increases by stop_loss_pct
            loss_threshold = entry_price * (1 + self.stop_loss_pct / 100)
            return current_price >= loss_threshold
        else:
            # For long positions, stop loss when price decreases by stop_loss_pct
            loss_threshold = entry_price * (1 - self.stop_loss_pct / 100)
            return current_price <= loss_threshold
    
    def check_take_profit(self, entry_price: float, 
                         current_price: float,
                         position_type: str = 'short') -> bool:
        """
        Check if take profit is hit.
        
        Args:
            entry_price: Entry premium
            current_price: Current premium
            position_type: 'short' or 'long'
            
        Returns:
            True if take profit is hit
        """
        if position_type.lower() == 'short':
            # For short positions, take profit when price decreases by take_profit_pct
            profit_threshold = entry_price * (1 - self.take_profit_pct / 100)
            return current_price <= profit_threshold
        else:
            # For long positions, take profit when price increases by take_profit_pct
            profit_threshold = entry_price * (1 + self.take_profit_pct / 100)
            return current_price >= profit_threshold
    
    def should_force_exit(self, current_time: pd.Timestamp) -> bool:
        """
        Check if position should be force exited based on time.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if should force exit
        """
        # Convert to time string for comparison
        current_time_str = current_time.strftime('%H:%M')
        return current_time_str >= self.force_exit_time
    
    def get_available_options(self, options_df: pd.DataFrame,
                            timestamp: pd.Timestamp,
                            expiry_date: pd.Timestamp,
                            option_type: str) -> pd.DataFrame:
        """
        Get all available options for given parameters.
        
        Args:
            options_df: Options data DataFrame
            timestamp: Trade timestamp
            expiry_date: Expiry date
            option_type: 'CE' or 'PE'
            
        Returns:
            Filtered DataFrame with available options
        """
        # Get data for the specific timestamp and expiry
        available_options = options_df[
            (options_df['datetime'] == timestamp) &
            (options_df['expiry_date'] == expiry_date) &
            (options_df['option_type'] == option_type)
        ].copy()
        
        # Filter out options with zero or NaN prices
        available_options = available_options[
            (available_options['close'] > 0) & 
            (~available_options['close'].isna())
        ]
        
        return available_options.sort_values('strike_price')
    
    def validate_trade_params(self, spot_price: float,
                            strike_price: float,
                            option_type: str,
                            entry_premium: float) -> Dict[str, Any]:
        """
        Validate trade parameters and calculate key metrics.
        
        Args:
            spot_price: Current spot price
            strike_price: Option strike price
            option_type: 'CE' or 'PE'
            entry_premium: Entry premium
            
        Returns:
            Dictionary with validation results and metrics
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Calculate moneyness
        if option_type.upper() == 'CE':
            moneyness = spot_price / strike_price
            itm = spot_price > strike_price
        else:  # PE
            moneyness = strike_price / spot_price
            itm = spot_price < strike_price
        
        validation['metrics'] = {
            'moneyness': moneyness,
            'is_itm': itm,
            'is_atm': abs(spot_price - strike_price) / spot_price < 0.02,  # Within 2%
            'premium_to_strike_ratio': entry_premium / strike_price if strike_price > 0 else 0
        }
        
        # Validation checks
        if entry_premium <= 0:
            validation['is_valid'] = False
            validation['warnings'].append("Entry premium must be positive")
        
        if strike_price <= 0:
            validation['is_valid'] = False
            validation['warnings'].append("Strike price must be positive")
        
        if spot_price <= 0:
            validation['is_valid'] = False
            validation['warnings'].append("Spot price must be positive")
        
        if option_type.upper() not in ['CE', 'PE']:
            validation['is_valid'] = False
            validation['warnings'].append("Option type must be 'CE' or 'PE'")
        
        # Warning for very high premium
        if entry_premium > strike_price * 0.1:  # Premium > 10% of strike
            validation['warnings'].append("Very high premium relative to strike price")
        
        # Warning for deep ITM/OTM options
        if abs(1 - moneyness) > 0.1:  # More than 10% away from ATM
            validation['warnings'].append("Option is significantly away from ATM")
        
        return validation
    
    def calculate_max_positions(self, capital: float, 
                              premium: float,
                              margin_multiplier: float = 1.0) -> int:
        """
        Calculate maximum number of positions based on available capital.
        
        Args:
            capital: Available capital
            premium: Option premium
            margin_multiplier: Margin requirement multiplier
            
        Returns:
            Maximum number of positions
        """
        if premium <= 0:
            return 0
        
        # For short options, consider margin requirement
        # Simplified margin calculation (actual SPAN margin is more complex)
        required_margin_per_lot = premium * self.lot_size * margin_multiplier
        
        max_positions = int(capital / required_margin_per_lot)
        return max(0, max_positions)
    
    def get_expiry_days(self, current_date: pd.Timestamp, 
                       expiry_date: pd.Timestamp) -> int:
        """
        Calculate days to expiry.
        
        Args:
            current_date: Current date
            expiry_date: Option expiry date
            
        Returns:
            Number of days to expiry
        """
        return (expiry_date.date() - current_date.date()).days
    
    def is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is within market hours.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if within market hours (9:15 to 15:30)
        """
        time_str = timestamp.strftime('%H:%M')
        return '09:15' <= time_str <= '15:30'
    
    def format_options_ticker(self, underlying: str, expiry_date: pd.Timestamp,
                            strike: float, option_type: str) -> str:
        """
        Format options ticker symbol.
        
        Args:
            underlying: Underlying symbol (e.g., 'NIFTY')
            expiry_date: Expiry date
            strike: Strike price
            option_type: 'CE' or 'PE'
            
        Returns:
            Formatted ticker symbol
        """
        expiry_str = expiry_date.strftime('%d%b%Y').upper()
        strike_str = f"{int(strike)}"
        
        return f"{underlying}{expiry_str}{strike_str}{option_type.upper()}"


def main():
    """Example usage of OptionsUtils class."""
    # Initialize options utils
    options_utils = OptionsUtils(lot_size=75)
    
    # Create sample options data
    sample_data = {
        'datetime': pd.date_range('2023-01-02 09:15', periods=10, freq='5T'),
        'strike_price': [18000] * 10,
        'option_type': ['PE'] * 10,
        'expiry_date': [pd.Timestamp('2023-01-05')] * 10,
        'open': np.random.uniform(100, 150, 10),
        'high': np.random.uniform(150, 200, 10),
        'low': np.random.uniform(50, 100, 10),
        'close': np.random.uniform(80, 140, 10),
        'volume': np.random.randint(100, 1000, 10),
        'oi': np.random.randint(10000, 50000, 10)
    }
    
    options_df = pd.DataFrame(sample_data)
    
    print("Options Utils Test Results:")
    print("="*50)
    
    # Test ATM strike selection
    spot_price = 18050
    available_strikes = [17800, 17900, 18000, 18100, 18200]
    atm_strike = options_utils.find_atm_strike(spot_price, available_strikes)
    print(f"Spot Price: {spot_price}")
    print(f"Available Strikes: {available_strikes}")
    print(f"ATM Strike: {atm_strike}")
    print()
    
    # Test nearest expiry
    current_time = pd.Timestamp('2023-01-02 10:00')
    available_expiries = [
        pd.Timestamp('2023-01-05'),
        pd.Timestamp('2023-01-12'),
        pd.Timestamp('2023-01-19')
    ]
    nearest_expiry = options_utils.find_nearest_expiry(current_time, available_expiries)
    print(f"Current Time: {current_time}")
    print(f"Available Expiries: {available_expiries}")
    print(f"Nearest Expiry: {nearest_expiry}")
    print()
    
    # Test options price retrieval
    test_timestamp = options_df['datetime'].iloc[0]
    option_price = options_utils.get_options_price(
        options_df, test_timestamp, 18000, 'PE', 
        pd.Timestamp('2023-01-05'), 'close'
    )
    print(f"Test Timestamp: {test_timestamp}")
    print(f"Retrieved Option Price: {option_price}")
    print()
    
    # Test P&L calculation
    entry_premium = 120.0
    exit_premium = 80.0
    pnl = options_utils.calculate_option_pnl(entry_premium, exit_premium, 'short')
    print(f"Entry Premium: {entry_premium}")
    print(f"Exit Premium: {exit_premium}")
    print(f"P&L (Short Position): ₹{pnl:,.2f}")
    print()
    
    # Test stop loss and take profit
    current_premium = 135.0
    sl_hit = options_utils.check_stop_loss(entry_premium, current_premium, 'short')
    tp_hit = options_utils.check_take_profit(entry_premium, current_premium, 'short')
    print(f"Current Premium: {current_premium}")
    print(f"Stop Loss Hit: {sl_hit}")
    print(f"Take Profit Hit: {tp_hit}")
    print()
    
    # Test trade validation
    validation = options_utils.validate_trade_params(
        spot_price=18050, strike_price=18000, 
        option_type='PE', entry_premium=120
    )
    print("Trade Validation:")
    print(f"Valid: {validation['is_valid']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Is ATM: {validation['metrics']['is_atm']}")
    print(f"Moneyness: {validation['metrics']['moneyness']:.4f}")
    print()
    
    # Test maximum positions calculation
    capital = 200000
    max_positions = options_utils.calculate_max_positions(capital, entry_premium, 1.0)
    print(f"Available Capital: ₹{capital:,}")
    print(f"Premium per lot: ₹{entry_premium * options_utils.lot_size:,}")
    print(f"Maximum Positions: {max_positions}")
    print()
    
    # Test market hours
    test_time = pd.Timestamp('2023-01-02 14:30')
    in_market_hours = options_utils.is_market_hours(test_time)
    print(f"Test Time: {test_time}")
    print(f"In Market Hours: {in_market_hours}")
    print()
    
    # Test ticker formatting
    ticker = options_utils.format_options_ticker(
        'NIFTY', pd.Timestamp('2023-01-05'), 18000, 'PE'
    )
    print(f"Formatted Ticker: {ticker}")
    print()
    
    # Test expiry days calculation
    expiry_days = options_utils.get_expiry_days(current_time, nearest_expiry)
    print(f"Days to Expiry: {expiry_days}")


if __name__ == "__main__":
    main()