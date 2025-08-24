"""
Main Orchestration Script for NIFTY Options Backtesting System

This script coordinates the complete pipeline:
1. Load and validate data
2. Add technical indicators  
3. Generate composite signals
4. Train ML models
5. Run backtesting
6. Generate comprehensive reports

Author: Options Backtesting System
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# Import our custom modules
from indicators import TechnicalIndicators
from signal_engine import SignalEngine  
from model import MLPredictor
from utils import OptionsUtils
from backtest import OptionsBacktester

warnings.filterwarnings('ignore')


class NiftyOptionsBacktester:
    """
    Main orchestrator for the NIFTY Options Backtesting System.
    
    This class coordinates the entire pipeline from data loading to report generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the main backtester with configuration.
        
        Args:
            config: Configuration dictionary with system parameters
        """
        # Default configuration
        self.config = {
            'data': {
                'spot_file': 'data/spot_with_signals_2023.csv',
                'options_file': 'data/options_data_2023.parquet',
                'timezone': 'Asia/Kolkata'
            },
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
            },
            'signals': {
                'buy_threshold': 0.6,
                'sell_threshold': 0.6,
                'optimize_thresholds': True,
                'custom_weights': None  # Will use default weights from SignalEngine
            },
            'ml': {
                'train_models': True,
                'models_to_train': ['linear', 'logistic', 'xgboost', 'random_forest', 'lstm'],
                'target_column': 'composite_signal',
                'test_size': 0.2,
                'cv_splits': 5
            },
            'backtest': {
                'initial_capital': 200000,
                'lot_size': 75,
                'stop_loss_pct': 1.5,
                'take_profit_pct': 3.0,
                'force_exit_time': '15:15'
            },
            'output': {
                'results_dir': 'results',
                'save_intermediate': True,
                'generate_plots': True
            }
        }
        
        # Update with user-provided config
        if config:
            self._update_config(self.config, config)
        
        # Initialize components
        self.spot_data = None
        self.options_data = None
        self.indicators = None
        self.signal_engine = None
        self.ml_predictor = None
        self.backtester = None
        
        # Results storage
        self.results = {}
        
    def _update_config(self, base_config: Dict, update_config: Dict) -> None:
        """Recursively update configuration dictionary."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def load_data(self) -> None:
        """Load and validate market data."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        try:
            # Load spot data
            spot_file = self.config['data']['spot_file']
            print(f"Loading spot data from: {spot_file}")
            
            if spot_file.endswith('.csv'):
                self.spot_data = pd.read_csv(spot_file)
            else:
                raise ValueError(f"Unsupported spot data format: {spot_file}")
            
            # Load options data
            options_file = self.config['data']['options_file']
            print(f"Loading options data from: {options_file}")
            
            if options_file.endswith('.parquet'):
                self.options_data = pd.read_parquet(options_file)
            elif options_file.endswith('.csv'):
                self.options_data = pd.read_csv(options_file)
            else:
                raise ValueError(f"Unsupported options data format: {options_file}")
            
            # Validate and process datetime columns
            self._process_datetime_columns()
            
            # Data validation
            self._validate_data()
            
            print(f"✓ Spot data loaded: {len(self.spot_data):,} rows")
            print(f"✓ Options data loaded: {len(self.options_data):,} rows")
            print(f"✓ Date range: {self.spot_data['datetime'].min()} to {self.spot_data['datetime'].max()}")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def _process_datetime_columns(self) -> None:
        """Process and standardize datetime columns."""
        timezone = self.config['data']['timezone']
        
        # Process spot data datetime
        if 'datetime' in self.spot_data.columns:
            self.spot_data['datetime'] = pd.to_datetime(self.spot_data['datetime'])
            if self.spot_data['datetime'].dt.tz is None:
                self.spot_data['datetime'] = self.spot_data['datetime'].dt.tz_localize(timezone)
            else:
                self.spot_data['datetime'] = self.spot_data['datetime'].dt.tz_convert(timezone)
        
        # Process options data datetime and expiry
        if 'datetime' in self.options_data.columns:
            self.options_data['datetime'] = pd.to_datetime(self.options_data['datetime'])
            if self.options_data['datetime'].dt.tz is None:
                self.options_data['datetime'] = self.options_data['datetime'].dt.tz_localize(timezone)
            else:
                self.options_data['datetime'] = self.options_data['datetime'].dt.tz_convert(timezone)
        
        if 'expiry_date' in self.options_data.columns:
            self.options_data['expiry_date'] = pd.to_datetime(self.options_data['expiry_date'])
        
        # Sort data by datetime
        self.spot_data = self.spot_data.sort_values('datetime').reset_index(drop=True)
        self.options_data = self.options_data.sort_values(['datetime', 'expiry_date', 'strike_price']).reset_index(drop=True)
    
    def _validate_data(self) -> None:
        """Validate loaded data for required columns and formats."""
        # Required columns for spot data
        required_spot_cols = ['datetime', 'open', 'high', 'low', 'close']
        missing_spot_cols = [col for col in required_spot_cols if col not in self.spot_data.columns]
        if missing_spot_cols:
            raise ValueError(f"Missing required spot data columns: {missing_spot_cols}")
        
        # Required columns for options data
        required_options_cols = ['datetime', 'strike_price', 'option_type', 'expiry_date', 'open', 'high', 'low', 'close']
        missing_options_cols = [col for col in required_options_cols if col not in self.options_data.columns]
        if missing_options_cols:
            raise ValueError(f"Missing required options data columns: {missing_options_cols}")
        
        # Validate option types
        valid_option_types = {'CE', 'PE'}
        actual_option_types = set(self.options_data['option_type'].unique())
        if not actual_option_types.issubset(valid_option_types):
            print(f"Warning: Found unexpected option types: {actual_option_types - valid_option_types}")
        
        # Check for missing values in critical columns
        critical_cols_spot = ['close']
        critical_cols_options = ['strike_price', 'close']
        
        for col in critical_cols_spot:
            missing_pct = self.spot_data[col].isna().sum() / len(self.spot_data) * 100
            if missing_pct > 5:  # More than 5% missing
                print(f"Warning: {col} has {missing_pct:.1f}% missing values in spot data")
        
        for col in critical_cols_options:
            missing_pct = self.options_data[col].isna().sum() / len(self.options_data) * 100
            if missing_pct > 10:  # More than 10% missing
                print(f"Warning: {col} has {missing_pct:.1f}% missing values in options data")
    
    def add_technical_indicators(self) -> None:
        """Add technical indicators to spot data."""
        print("\n" + "=" * 60)
        print("ADDING TECHNICAL INDICATORS")
        print("=" * 60)
        
        try:
            # Initialize technical indicators
            self.indicators = TechnicalIndicators(self.spot_data)
            
            # Add indicators with custom parameters
            ind_config = self.config['indicators']
            
            enhanced_data = (self.indicators
                           .add_macd(ind_config['macd_fast'], ind_config['macd_slow'], ind_config['macd_signal'])
                           .add_rsi(ind_config['rsi_period'])
                           .add_bollinger_bands(ind_config['bb_period'], ind_config['bb_std'])
                           .add_atr(14)  # ATR needed for SuperTrend
                           .add_supertrend(ind_config['supertrend_period'], ind_config['supertrend_multiplier'])
                           .add_adx(ind_config['adx_period'])
                           .add_stochastic(ind_config['stoch_k'], ind_config['stoch_d'])
                           .add_ema_crossover(ind_config['ema_fast'], ind_config['ema_slow'])
                           .get_dataframe())
            
            self.spot_data = enhanced_data
            
            # Get indicator summary
            indicator_summary = self.indicators.get_signal_summary()
            
            print(f"✓ Technical indicators added successfully")
            print(f"✓ Latest RSI: {indicator_summary['values']['rsi']:.2f}")
            print(f"✓ Latest ADX: {indicator_summary['values']['adx']:.2f}")
            print(f"✓ Active signals: {sum(indicator_summary['signals'].values())} out of {len(indicator_summary['signals'])}")
            
            # Save intermediate results if configured
            if self.config['output']['save_intermediate']:
                output_file = os.path.join(self.config['output']['results_dir'], 'spot_with_indicators.csv')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                self.spot_data.to_csv(output_file, index=False)
                print(f"✓ Spot data with indicators saved to: {output_file}")
                
        except Exception as e:
            print(f"✗ Error adding technical indicators: {e}")
            raise
    
    def generate_signals(self) -> None:
        """Generate composite signals using weighted voting."""
        print("\n" + "=" * 60)
        print("GENERATING COMPOSITE SIGNALS")
        print("=" * 60)
        
        try:
            # Initialize signal engine
            signal_config = self.config['signals']
            
            self.signal_engine = SignalEngine(
                buy_threshold=signal_config['buy_threshold'],
                sell_threshold=signal_config['sell_threshold'],
                signal_weights=signal_config['custom_weights']
            )
            
            # Generate composite signals
            self.spot_data = self.signal_engine.generate_composite_signals(self.spot_data)
            
            # Optimize thresholds if configured
            if signal_config['optimize_thresholds']:
                print("Optimizing signal thresholds...")
                optimal_thresholds = self.signal_engine.optimize_thresholds(self.spot_data)
                print(f"✓ Optimal buy threshold: {optimal_thresholds['buy_threshold']:.2f}")
                print(f"✓ Optimal sell threshold: {optimal_thresholds['sell_threshold']:.2f}")
                
                # Update thresholds and regenerate signals
                self.signal_engine.buy_threshold = optimal_thresholds['buy_threshold']
                self.signal_engine.sell_threshold = optimal_thresholds['sell_threshold']
                self.spot_data = self.signal_engine.generate_composite_signals(self.spot_data)
            
            # Get signal performance
            signal_performance = self.signal_engine.get_signal_performance(self.spot_data)
            
            print(f"✓ Composite signals generated")
            print(f"✓ Buy signals: {signal_performance['signal_distribution']['buy_count']} ({signal_performance['signal_distribution']['buy_percentage']:.1f}%)")
            print(f"✓ Sell signals: {signal_performance['signal_distribution']['sell_count']} ({signal_performance['signal_distribution']['sell_percentage']:.1f}%)")
            print(f"✓ Hold signals: {signal_performance['signal_distribution']['hold_count']} ({signal_performance['signal_distribution']['hold_percentage']:.1f}%)")
            
            # Store results
            self.results['signal_performance'] = signal_performance
            
            # Save intermediate results
            if self.config['output']['save_intermediate']:
                output_file = os.path.join(self.config['output']['results_dir'], 'spot_with_signals.csv')
                self.spot_data.to_csv(output_file, index=False)
                print(f"✓ Spot data with composite signals saved to: {output_file}")
                
        except Exception as e:
            print(f"✗ Error generating signals: {e}")
            raise
    
    def train_ml_models(self) -> None:
        """Train machine learning models."""
        if not self.config['ml']['train_models']:
            print("\nSkipping ML model training (disabled in config)")
            return
            
        print("\n" + "=" * 60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("=" * 60)
        
        try:
            # Initialize ML predictor
            self.ml_predictor = MLPredictor(random_state=42)
            
            # Prepare target variable
            target_col = self.config['ml']['target_column']
            
            if target_col not in self.spot_data.columns:
                print(f"Warning: Target column '{target_col}' not found. Using 'composite_signal'")
                target_col = 'composite_signal'
            
            # Train models
            print(f"Training models with target: {target_col}")
            ml_results = self.ml_predictor.train_all_models(self.spot_data, target_col)
            
            # Evaluate models
            model_evaluation = self.ml_predictor.evaluate_models(ml_results)
            print("\n✓ Model Performance Summary:")
            print(model_evaluation.to_string(index=False))
            
            # Get best model
            best_model_name, best_model_results = self.ml_predictor.get_best_model(ml_results)
            print(f"\n✓ Best Model: {best_model_name}")
            
            if self.ml_predictor.model_type == 'classification':
                print(f"✓ Best Accuracy: {best_model_results.get('accuracy', 0):.4f}")
            else:
                print(f"✓ Best R² Score: {best_model_results.get('r2', 0):.4f}")
            
            # Generate predictions for backtesting
            predictions = self.ml_predictor.predict(self.spot_data)
            self.spot_data['ml_predictions'] = predictions
            
            # Store results
            self.results['ml_results'] = ml_results
            self.results['model_evaluation'] = model_evaluation
            self.results['best_model'] = {'name': best_model_name, 'results': best_model_results}
            
            # Save model evaluation
            if self.config['output']['save_intermediate']:
                output_file = os.path.join(self.config['output']['results_dir'], 'model_evaluation.csv')
                model_evaluation.to_csv(output_file, index=False)
                print(f"✓ Model evaluation saved to: {output_file}")
                
        except Exception as e:
            print(f"✗ Error training ML models: {e}")
            print("Continuing without ML predictions...")
            self.config['ml']['train_models'] = False
    
    def run_backtest(self) -> None:
        """Run the options backtesting."""
        print("\n" + "=" * 60)
        print("RUNNING OPTIONS BACKTEST")
        print("=" * 60)
        
        try:
            # Initialize backtester
            backtest_config = self.config['backtest']
            
            self.backtester = OptionsBacktester(
                initial_capital=backtest_config['initial_capital'],
                lot_size=backtest_config['lot_size'],
                stop_loss_pct=backtest_config['stop_loss_pct'],
                take_profit_pct=backtest_config['take_profit_pct'],
                force_exit_time=backtest_config['force_exit_time']
            )
            
            # Load data into backtester
            self.backtester.load_data(self.spot_data, self.options_data)
            
            # Run backtest
            print("Starting backtest execution...")
            backtest_results = self.backtester.run_backtest()
            
            # Display key results
            metrics = backtest_results['performance_metrics']
            print(f"\n✓ Backtest completed successfully!")
            print(f"✓ Total Return: {metrics['total_return']:.2f}%")
            print(f"✓ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"✓ Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"✓ Win Rate: {metrics['win_rate']:.1f}%")
            print(f"✓ Total Trades: {metrics['total_trades']}")
            print(f"✓ Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"✓ Final Capital: ₹{metrics['final_capital']:,.2f}")
            
            # Store results
            self.results['backtest_results'] = backtest_results
            
        except Exception as e:
            print(f"✗ Error running backtest: {e}")
            raise
    
    def generate_reports(self) -> None:
        """Generate comprehensive reports and visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING REPORTS")
        print("=" * 60)
        
        try:
            results_dir = self.config['output']['results_dir']
            os.makedirs(results_dir, exist_ok=True)
            
            # Save backtest results (includes plots)
            if 'backtest_results' in self.results:
                self.backtester.save_results(self.results['backtest_results'], results_dir)
                print(f"✓ Backtest results saved to: {results_dir}")
            
            # Generate summary report
            self._generate_summary_report(results_dir)
            
            # Generate detailed analysis if ML models were trained
            if self.config['ml']['train_models'] and 'ml_results' in self.results:
                self._generate_ml_report(results_dir)
            
            print(f"✓ All reports generated in: {results_dir}")
            
        except Exception as e:
            print(f"✗ Error generating reports: {e}")
            raise
    
    def _generate_summary_report(self, output_dir: str) -> None:
        """Generate a comprehensive summary report."""
        summary_data = {
            'section': [],
            'metric': [],
            'value': []
        }
        
        # Add configuration summary
        summary_data['section'].extend(['Configuration'] * 5)
        summary_data['metric'].extend(['Initial Capital', 'Stop Loss %', 'Take Profit %', 'Force Exit Time', 'Lot Size'])
        summary_data['value'].extend([
            f"₹{self.config['backtest']['initial_capital']:,}",
            f"{self.config['backtest']['stop_loss_pct']}%",
            f"{self.config['backtest']['take_profit_pct']}%", 
            self.config['backtest']['force_exit_time'],
            str(self.config['backtest']['lot_size'])
        ])
        
        # Add data summary
        summary_data['section'].extend(['Data'] * 4)
        summary_data['metric'].extend(['Spot Data Rows', 'Options Data Rows', 'Date Range Start', 'Date Range End'])
        summary_data['value'].extend([
            f"{len(self.spot_data):,}",
            f"{len(self.options_data):,}",
            str(self.spot_data['datetime'].min().date()),
            str(self.spot_data['datetime'].max().date())
        ])
        
        # Add signal summary
        if 'signal_performance' in self.results:
            perf = self.results['signal_performance']
            summary_data['section'].extend(['Signals'] * 6)
            summary_data['metric'].extend(['Buy Signals', 'Sell Signals', 'Hold Signals', 'Buy %', 'Sell %', 'Hold %'])
            summary_data['value'].extend([
                str(perf['signal_distribution']['buy_count']),
                str(perf['signal_distribution']['sell_count']),
                str(perf['signal_distribution']['hold_count']),
                f"{perf['signal_distribution']['buy_percentage']:.1f}%",
                f"{perf['signal_distribution']['sell_percentage']:.1f}%",
                f"{perf['signal_distribution']['hold_percentage']:.1f}%"
            ])
        
        # Add backtest performance
        if 'backtest_results' in self.results:
            metrics = self.results['backtest_results']['performance_metrics']
            summary_data['section'].extend(['Performance'] * 8)
            summary_data['metric'].extend([
                'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate',
                'Total Trades', 'Profit Factor', 'Final Capital', 'Total P&L'
            ])
            summary_data['value'].extend([
                f"{metrics['total_return']:.2f}%",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['max_drawdown']:.2f}%",
                f"{metrics['win_rate']:.1f}%",
                str(metrics['total_trades']),
                f"{metrics['profit_factor']:.2f}",
                f"₹{metrics['final_capital']:,.2f}",
                f"₹{metrics['total_pnl']:,.2f}"
            ])
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'summary_report.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Summary report saved to: {summary_file}")
    
    def _generate_ml_report(self, output_dir: str) -> None:
        """Generate ML model analysis report."""
        if 'model_evaluation' in self.results:
            ml_report_file = os.path.join(output_dir, 'ml_model_comparison.csv')
            self.results['model_evaluation'].to_csv(ml_report_file, index=False)
            print(f"✓ ML model comparison saved to: {ml_report_file}")
        
        # Feature importance for best model
        if 'best_model' in self.results:
            best_model_results = self.results['best_model']['results']
            if 'feature_importance' in best_model_results:
                importance_data = []
                for feature, importance in sorted(
                    best_model_results['feature_importance'].items(),
                    key=lambda x: x[1], reverse=True
                ):
                    importance_data.append({'feature': feature, 'importance': importance})
                
                importance_df = pd.DataFrame(importance_data)
                importance_file = os.path.join(output_dir, 'feature_importance.csv')
                importance_df.to_csv(importance_file, index=False)
                print(f"✓ Feature importance saved to: {importance_file}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete backtesting pipeline."""
        start_time = datetime.now()
        
        print("🚀 Starting NIFTY Options Backtesting Pipeline")
        print(f"Start Time: {start_time}")
        print("=" * 60)
        
        try:
            # Execute pipeline steps
            self.load_data()
            self.add_technical_indicators() 
            self.generate_signals()
            self.train_ml_models()
            self.run_backtest()
            self.generate_reports()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Total Runtime: {duration:.1f} seconds")
            print(f"End Time: {end_time}")
            
            # Final summary
            if 'backtest_results' in self.results:
                metrics = self.results['backtest_results']['performance_metrics']
                print(f"\n📈 FINAL RESULTS:")
                print(f"   Total Return: {metrics['total_return']:.2f}%")
                print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
                print(f"   Win Rate: {metrics['win_rate']:.1f}%")
                print(f"   Final Capital: ₹{metrics['final_capital']:,.2f}")
            
            print(f"\n📁 Results saved in: {self.config['output']['results_dir']}")
            print("   - equity_curve.png")
            print("   - drawdown.png") 
            print("   - metrics.csv")
            print("   - trades.csv")
            print("   - summary_report.csv")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration file."""
    return {
        'data': {
            'spot_file': 'data/spot_with_signals_2023.csv',
            'options_file': 'data/options_data_2023.parquet',
            'timezone': 'Asia/Kolkata'
        },
        'backtest': {
            'initial_capital': 200000,
            'stop_loss_pct': 1.5,
            'take_profit_pct': 3.0
        },
        'signals': {
            'buy_threshold': 0.6,
            'sell_threshold': 0.6,
            'optimize_thresholds': True
        },
        'ml': {
            'train_models': True
        },
        'output': {
            'results_dir': 'results',
            'generate_plots': True
        }
    }


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='NIFTY Options Backtesting System')
    parser.add_argument('--spot-file', type=str, default='data/spot_with_signals_2023.csv',
                       help='Path to spot data CSV file')
    parser.add_argument('--options-file', type=str, default='data/options_data_2023.parquet', 
                       help='Path to options data parquet/CSV file')
    parser.add_argument('--capital', type=float, default=200000,
                       help='Initial capital in rupees')
    parser.add_argument('--stop-loss', type=float, default=1.5,
                       help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=3.0,
                       help='Take profit percentage')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--no-ml', action='store_true',
                       help='Disable ML model training')
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_sample_config()
    
    # Override config with command line arguments
    config['data']['spot_file'] = args.spot_file
    config['data']['options_file'] = args.options_file
    config['backtest']['initial_capital'] = args.capital
    config['backtest']['stop_loss_pct'] = args.stop_loss
    config['backtest']['take_profit_pct'] = args.take_profit
    config['output']['results_dir'] = args.results_dir
    config['ml']['train_models'] = not args.no_ml
    
    # Validate input files exist
    if not os.path.exists(args.spot_file):
        print(f"❌ Error: Spot data file not found: {args.spot_file}")
        sys.exit(1)
    
    if not os.path.exists(args.options_file):
        print(f"❌ Error: Options data file not found: {args.options_file}")
        sys.exit(1)
    
    try:
        # Initialize and run backtester
        backtester = NiftyOptionsBacktester(config)
        results = backtester.run_complete_pipeline()
        
        print("\n🎯 EXECUTION COMPLETED SUCCESSFULLY!")
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()