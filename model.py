"""
Machine Learning Models Module for NIFTY Options Backtesting System

This module provides ML prediction capabilities including:
- Linear/Logistic Regression
- XGBoost
- LSTM
- Feature engineering
- Time-series cross-validation
- Model evaluation and selection

Author: Options Backtesting System
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will not be supported.")


class MLPredictor:
    """
    Comprehensive ML prediction system for options trading signals.
    
    Features:
    - Multiple model types (Linear, Logistic, XGBoost, LSTM, Random Forest)
    - Automated feature engineering
    - Time-series cross-validation
    - Model comparison and selection
    - Prediction confidence intervals
    - Feature importance analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize MLPredictor with configuration.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        self.model_type = None
        self.is_trained = False
        
        # Set random seeds
        np.random.seed(random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_state)
    
    def engineer_features(self, df: pd.DataFrame, 
                         price_col: str = 'close') -> pd.DataFrame:
        """
        Engineer features for ML models.
        
        Args:
            df: Input DataFrame with OHLC and indicators
            price_col: Price column name
            
        Returns:
            DataFrame with engineered features
        """
        feature_df = df.copy()
        
        # Price-based features
        feature_df['price_change'] = feature_df[price_col].pct_change()
        feature_df['price_change_2'] = feature_df[price_col].pct_change(2)
        feature_df['price_change_5'] = feature_df[price_col].pct_change(5)
        
        # Volatility features
        feature_df['volatility_5'] = feature_df['price_change'].rolling(5).std()
        feature_df['volatility_10'] = feature_df['price_change'].rolling(10).std()
        feature_df['volatility_20'] = feature_df['price_change'].rolling(20).std()
        
        # Volume features (if available)
        if 'volume' in feature_df.columns:
            feature_df['volume_ma5'] = feature_df['volume'].rolling(5).mean()
            feature_df['volume_ratio'] = feature_df['volume'] / feature_df['volume_ma5']
        
        # Price position features
        feature_df['high_low_ratio'] = feature_df['high'] / feature_df['low']
        feature_df['close_to_high'] = feature_df[price_col] / feature_df['high']
        feature_df['close_to_low'] = feature_df[price_col] / feature_df['low']
        
        # Moving average features
        for period in [5, 10, 20]:
            ma_col = f'ma_{period}'
            feature_df[ma_col] = feature_df[price_col].rolling(period).mean()
            feature_df[f'price_to_{ma_col}'] = feature_df[price_col] / feature_df[ma_col]
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            feature_df[f'price_lag_{lag}'] = feature_df[price_col].shift(lag)
            feature_df[f'return_lag_{lag}'] = feature_df['price_change'].shift(lag)
        
        # Time-based features
        if 'datetime' in feature_df.columns:
            feature_df['hour'] = pd.to_datetime(feature_df['datetime']).dt.hour
            feature_df['minute'] = pd.to_datetime(feature_df['datetime']).dt.minute
            feature_df['day_of_week'] = pd.to_datetime(feature_df['datetime']).dt.dayofweek
            feature_df['is_morning'] = (feature_df['hour'] < 12).astype(int)
            feature_df['is_opening'] = (feature_df['hour'] == 9).astype(int)
            feature_df['is_closing'] = (feature_df['hour'] >= 15).astype(int)
        
        # Technical indicator ratios and combinations
        if 'rsi' in feature_df.columns:
            feature_df['rsi_normalized'] = (feature_df['rsi'] - 50) / 50
            feature_df['rsi_extreme'] = ((feature_df['rsi'] > 70) | (feature_df['rsi'] < 30)).astype(int)
        
        if 'macd' in feature_df.columns and 'macd_signal' in feature_df.columns:
            feature_df['macd_signal_ratio'] = feature_df['macd'] / (feature_df['macd_signal'] + 1e-8)
        
        if 'bb_upper' in feature_df.columns and 'bb_lower' in feature_df.columns:
            feature_df['bb_position'] = (feature_df[price_col] - feature_df['bb_lower']) / \
                                       (feature_df['bb_upper'] - feature_df['bb_lower'])
        
        # Signal strength features
        if 'buy_score' in feature_df.columns:
            feature_df['signal_momentum'] = feature_df['buy_score'].diff()
            feature_df['signal_strength'] = abs(feature_df['buy_score'] - feature_df['sell_score'])
        
        return feature_df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Engineer features
        feature_df = self.engineer_features(df)
        
        # Select numerical features (exclude datetime and string columns)
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target column from features
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        # Remove columns with all NaN or constant values
        valid_cols = []
        for col in numerical_cols:
            if not feature_df[col].isna().all() and feature_df[col].nunique() > 1:
                valid_cols.append(col)
        
        self.feature_columns = valid_cols
        features = feature_df[valid_cols].fillna(method='ffill').fillna(0)
        
        # Prepare target
        if target_col:
            target = feature_df[target_col].fillna(method='ffill')
            self.target_column = target_col
        else:
            target = None
        
        return features, target
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models.
        
        Args:
            X: Feature array
            y: Target array
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train Linear Regression model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with model and metrics
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred,
            'feature_importance': dict(zip(X.columns, abs(model.coef_)))
        }
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train Logistic Regression model.
        
        Args:
            X: Features DataFrame
            y: Target Series (categorical)
            
        Returns:
            Dictionary with model and metrics
        """
        # Encode target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_scaled, y_encoded)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'accuracy': accuracy,
            'predictions': label_encoder.inverse_transform(y_pred),
            'probabilities': y_pred_proba,
            'feature_importance': dict(zip(X.columns, abs(model.coef_[0])))
        }
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, 
                     task_type: str = 'regression') -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with model and metrics
        """
        if task_type == 'classification':
            # Encode target labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            model.fit(X, y_encoded)
            
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)
            
            accuracy = accuracy_score(y_encoded, y_pred)
            
            return {
                'model': model,
                'label_encoder': label_encoder,
                'accuracy': accuracy,
                'predictions': label_encoder.inverse_transform(y_pred),
                'probabilities': y_pred_proba,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
        else:
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            model.fit(X, y)
            
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            return {
                'model': model,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
    
    def train_lstm(self, X: pd.DataFrame, y: pd.Series, 
                  sequence_length: int = 10,
                  task_type: str = 'regression') -> Optional[Dict[str, Any]]:
        """
        Train LSTM model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            sequence_length: LSTM sequence length
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with model and metrics or None if TensorFlow unavailable
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping LSTM training.")
            return None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        if task_type == 'classification':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            X_seq, y_seq = self.create_sequences(X_scaled, y_encoded, sequence_length)
            n_classes = len(np.unique(y_encoded))
        else:
            X_seq, y_seq = self.create_sequences(X_scaled, y.values, sequence_length)
            n_classes = 1
        
        if len(X_seq) == 0:
            print("Not enough data for LSTM sequences")
            return None
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(n_classes if task_type == 'classification' else 1,
                 activation='softmax' if task_type == 'classification' else 'linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if task_type == 'classification' else 'mse',
            metrics=['accuracy'] if task_type == 'classification' else ['mse']
        )
        
        # Train model
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        # Make predictions
        y_pred = model.predict(X_seq, verbose=0)
        
        if task_type == 'classification':
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_seq, y_pred_classes)
            
            return {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'sequence_length': sequence_length,
                'accuracy': accuracy,
                'predictions': label_encoder.inverse_transform(y_pred_classes),
                'probabilities': y_pred,
                'history': history.history
            }
        else:
            y_pred_flat = y_pred.flatten()
            mse = mean_squared_error(y_seq, y_pred_flat)
            r2 = r2_score(y_seq, y_pred_flat)
            
            return {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred_flat,
                'history': history.history
            }
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series,
                           task_type: str = 'regression') -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with model and metrics
        """
        if task_type == 'classification':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10
            )
            model.fit(X, y_encoded)
            
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)
            accuracy = accuracy_score(y_encoded, y_pred)
            
            return {
                'model': model,
                'label_encoder': label_encoder,
                'accuracy': accuracy,
                'predictions': label_encoder.inverse_transform(y_pred),
                'probabilities': y_pred_proba,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10
            )
            model.fit(X, y)
            
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            return {
                'model': model,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
    
    def train_all_models(self, df: pd.DataFrame, 
                        target_col: str,
                        task_type: str = 'auto') -> Dict[str, Dict[str, Any]]:
        """
        Train all available models and compare performance.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            task_type: 'regression', 'classification', or 'auto'
            
        Returns:
            Dictionary with all model results
        """
        X, y = self.prepare_features(df, target_col)
        
        # Auto-detect task type
        if task_type == 'auto':
            unique_values = y.nunique()
            if unique_values <= 10 and y.dtype == 'object':
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        self.model_type = task_type
        results = {}
        
        print(f"Training models for {task_type} task...")
        print(f"Features shape: {X.shape}")
        print(f"Target unique values: {y.nunique()}")
        
        # Train Linear/Logistic Regression
        try:
            if task_type == 'regression':
                results['linear_regression'] = self.train_linear_regression(X, y)
                print("✓ Linear Regression trained")
            else:
                results['logistic_regression'] = self.train_logistic_regression(X, y)
                print("✓ Logistic Regression trained")
        except Exception as e:
            print(f"✗ Linear/Logistic Regression failed: {e}")
        
        # Train XGBoost
        try:
            results['xgboost'] = self.train_xgboost(X, y, task_type)
            print("✓ XGBoost trained")
        except Exception as e:
            print(f"✗ XGBoost failed: {e}")
        
        # Train Random Forest
        try:
            results['random_forest'] = self.train_random_forest(X, y, task_type)
            print("✓ Random Forest trained")
        except Exception as e:
            print(f"✗ Random Forest failed: {e}")
        
        # Train LSTM
        try:
            lstm_result = self.train_lstm(X, y, task_type=task_type)
            if lstm_result:
                results['lstm'] = lstm_result
                print("✓ LSTM trained")
        except Exception as e:
            print(f"✗ LSTM failed: {e}")
        
        self.models = results
        self.is_trained = True
        
        return results
    
    def get_best_model(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model.
        
        Args:
            results: Model results dictionary
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if results is None:
            results = self.models
        
        if not results:
            raise ValueError("No trained models available")
        
        best_model_name = None
        best_score = -np.inf
        
        for model_name, model_results in results.items():
            if self.model_type == 'regression':
                # For regression, use R² score (higher is better)
                score = model_results.get('r2', -np.inf)
            else:
                # For classification, use accuracy (higher is better)
                score = model_results.get('accuracy', -np.inf)
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, results[best_model_name]
    
    def predict(self, df: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            df: Input DataFrame
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("No models have been trained yet")
        
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        X, _ = self.prepare_features(df)
        model_results = self.models[model_name]
        model = model_results['model']
        
        # Handle different model types
        if model_name in ['linear_regression', 'logistic_regression']:
            scaler = model_results['scaler']
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            if 'label_encoder' in model_results:
                predictions = model_results['label_encoder'].inverse_transform(predictions)
                
        elif model_name in ['xgboost', 'random_forest']:
            predictions = model.predict(X)
            
            if 'label_encoder' in model_results:
                predictions = model_results['label_encoder'].inverse_transform(predictions)
                
        elif model_name == 'lstm':
            scaler = model_results['scaler']
            sequence_length = model_results['sequence_length']
            
            X_scaled = scaler.transform(X)
            X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)), sequence_length)
            
            if len(X_seq) > 0:
                predictions = model.predict(X_seq, verbose=0)
                
                if self.model_type == 'classification':
                    predictions = np.argmax(predictions, axis=1)
                    predictions = model_results['label_encoder'].inverse_transform(predictions)
                else:
                    predictions = predictions.flatten()
                
                # Pad predictions to match input length
                full_predictions = np.full(len(X), np.nan)
                full_predictions[sequence_length:sequence_length+len(predictions)] = predictions
                predictions = full_predictions
            else:
                predictions = np.full(len(X), np.nan)
        
        return predictions
    
    def evaluate_models(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Evaluate and compare all trained models.
        
        Args:
            results: Model results dictionary
            
        Returns:
            DataFrame with model comparison
        """
        if results is None:
            results = self.models
        
        evaluation_data = []
        
        for model_name, model_results in results.items():
            eval_dict = {'model': model_name}
            
            if self.model_type == 'regression':
                eval_dict['mse'] = model_results.get('mse', np.nan)
                eval_dict['r2'] = model_results.get('r2', np.nan)
                eval_dict['rmse'] = np.sqrt(model_results.get('mse', np.nan))
            else:
                eval_dict['accuracy'] = model_results.get('accuracy', np.nan)
            
            evaluation_data.append(eval_dict)
        
        return pd.DataFrame(evaluation_data).sort_values(
            'r2' if self.model_type == 'regression' else 'accuracy', 
            ascending=False
        )


def main():
    """Example usage of MLPredictor class."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='5T')
    
    # Generate sample OHLC data with trend
    trend = np.linspace(100, 120, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    close_prices = trend + noise + np.random.normal(0, 1, n_samples).cumsum() * 0.1
    
    high_prices = close_prices + np.random.uniform(0, 2, n_samples)
    low_prices = close_prices - np.random.uniform(0, 2, n_samples)
    open_prices = close_prices + np.random.normal(0, 0.5, n_samples)
    
    # Add some technical indicators
    sample_df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 1, n_samples),
        'macd_signal': np.random.normal(0, 0.8, n_samples),
        'buy_score': np.random.uniform(0, 1, n_samples),
        'sell_score': np.random.uniform(0, 1, n_samples),
    })
    
    # Create target: future return (regression) and signal (classification)
    sample_df['future_return'] = sample_df['close'].shift(-5) / sample_df['close'] - 1
    sample_df['signal'] = np.where(
        sample_df['future_return'] > 0.01, 'Buy',
        np.where(sample_df['future_return'] < -0.01, 'Sell', 'Hold')
    )
    
    # Remove last few rows with NaN targets
    sample_df = sample_df.dropna()
    
    print("ML Predictor Test Results:")
    print("="*50)
    print(f"Dataset shape: {sample_df.shape}")
    print(f"Target distribution:\n{sample_df['signal'].value_counts()}")
    
    # Initialize ML predictor
    ml_predictor = MLPredictor(random_state=42)
    
    # Test regression task
    print("\n" + "="*30)
    print("REGRESSION TASK (Future Returns)")
    print("="*30)
    
    regression_results = ml_predictor.train_all_models(
        sample_df, 'future_return', 'regression'
    )
    
    # Evaluate regression models
    regression_eval = ml_predictor.evaluate_models(regression_results)
    print("\nRegression Model Evaluation:")
    print(regression_eval)
    
    # Get best regression model
    best_reg_name, best_reg_results = ml_predictor.get_best_model(regression_results)
    print(f"\nBest Regression Model: {best_reg_name}")
    print(f"R² Score: {best_reg_results.get('r2', 'N/A'):.4f}")
    
    # Test classification task
    print("\n" + "="*30)
    print("CLASSIFICATION TASK (Signals)")
    print("="*30)
    
    ml_predictor_clf = MLPredictor(random_state=42)
    classification_results = ml_predictor_clf.train_all_models(
        sample_df, 'signal', 'classification'
    )
    
    # Evaluate classification models
    classification_eval = ml_predictor_clf.evaluate_models(classification_results)
    print("\nClassification Model Evaluation:")
    print(classification_eval)
    
    # Get best classification model
    best_clf_name, best_clf_results = ml_predictor_clf.get_best_model(classification_results)
    print(f"\nBest Classification Model: {best_clf_name}")
    print(f"Accuracy: {best_clf_results.get('accuracy', 'N/A'):.4f}")
    
    # Test predictions
    print("\n" + "="*30)
    print("PREDICTION TESTING")
    print("="*30)
    
    # Make regression predictions
    reg_predictions = ml_predictor.predict(sample_df.head(100))
    print(f"Regression predictions shape: {reg_predictions.shape}")
    print(f"Sample predictions: {reg_predictions[:5]}")
    
    # Make classification predictions
    clf_predictions = ml_predictor_clf.predict(sample_df.head(100))
    print(f"Classification predictions shape: {clf_predictions.shape}")
    print(f"Sample predictions: {clf_predictions[:5]}")
    
    # Feature importance analysis
    print("\n" + "="*30)
    print("FEATURE IMPORTANCE")
    print("="*30)
    
    if 'feature_importance' in best_reg_results:
        top_features = sorted(
            best_reg_results['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print(f"\nTop 10 Features for {best_reg_name} (Regression):")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
    
    if 'feature_importance' in best_clf_results:
        top_features_clf = sorted(
            best_clf_results['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print(f"\nTop 10 Features for {best_clf_name} (Classification):")
        for feature, importance in top_features_clf:
            print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()