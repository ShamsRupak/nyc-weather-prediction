"""
Advanced NYC Weather Prediction Project
======================================
Enhanced ML pipeline with XGBoost, hyperparameter tuning, cross-validation,
and advanced feature engineering for weather prediction in New York City.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, install if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherDataGenerator:
    """Generate realistic mock weather data with advanced patterns"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Generate realistic weather data with complex patterns"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Enhanced seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Base temperature with multiple harmonics for realistic patterns
        base_temp = (50 + 
                    25 * np.sin(2 * np.pi * (day_of_year - 80) / 365) +  # Annual cycle
                    3 * np.sin(4 * np.pi * day_of_year / 365) +           # Semi-annual
                    1 * np.sin(8 * np.pi * day_of_year / 365))            # Quarterly
        
        # Add year-to-year variation and trend
        years = np.array([d.year for d in dates])
        climate_trend = 0.1 * (years - 2020)  # Slight warming trend
        yearly_variation = 2 * np.sin(2 * np.pi * years / 11)  # Solar cycle influence
        
        # Temperature with noise and patterns
        temperature = (base_temp + climate_trend + yearly_variation + 
                      np.random.normal(0, 6, n_days))
        
        # Enhanced humidity model
        humidity = (70 - 0.4 * (temperature - 50) + 
                   15 * np.sin(2 * np.pi * (day_of_year - 120) / 365) +
                   np.random.normal(0, 8, n_days))
        humidity = np.clip(humidity, 15, 98)
        
        # Atmospheric pressure with weather patterns
        pressure = (1013 + 
                   5 * np.sin(2 * np.pi * day_of_year / 365) +
                   3 * np.random.randn(n_days))  # Weather systems
        
        # Wind speed with seasonal and random components
        wind_speed = (12 + 
                     6 * np.sin(2 * np.pi * (day_of_year + 180) / 365) +
                     np.random.exponential(4, n_days))
        wind_speed = np.clip(wind_speed, 0, 35)
        
        # Precipitation with seasonal patterns and clustering
        precip_base = np.random.exponential(0.08, n_days)
        seasonal_precip = 1 + 0.7 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        precipitation = precip_base * seasonal_precip
        
        # Add storm events (clustering)
        storm_events = np.random.random(n_days) < 0.05
        precipitation[storm_events] *= np.random.uniform(3, 8, storm_events.sum())
        precipitation = np.clip(precipitation, 0, 6)
        
        return pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })

class AdvancedFeatureEngineer:
    """Create advanced features for weather prediction"""
    
    @staticmethod
    def create_features(data):
        """Generate comprehensive feature set"""
        df = data.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Seasonal features
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Lag features (previous days)
        for col in ['humidity', 'pressure', 'wind_speed', 'precipitation']:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_lag7'] = df[col].shift(7)
        
        # Rolling statistics
        window_sizes = [3, 7, 14]
        for window in window_sizes:
            for col in ['humidity', 'pressure', 'wind_speed']:
                df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window=window).std()
        
        # Weather indices
        df['heat_index'] = df['temperature'] + 0.5 * df['humidity'] - 40
        df['wind_chill'] = 35.74 + 0.6215 * df['temperature'] - 35.75 * (df['wind_speed'] ** 0.16)
        df['comfort_index'] = df['temperature'] - 0.1 * df['humidity'] - 0.2 * df['wind_speed']
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        df['pressure_wind_interaction'] = df['pressure'] * df['wind_speed']
        
        return df

class AdvancedWeatherPredictor:
    """Advanced weather prediction with multiple models and tuning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.results = {}
    
    def prepare_features(self, data):
        """Prepare feature matrix for modeling"""
        # Select numeric features (excluding date and categorical)
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'temperature' in numeric_features:
            numeric_features.remove('temperature')
        
        # Remove features with too many NaNs
        valid_features = []
        for col in numeric_features:
            if data[col].isna().sum() / len(data) < 0.1:  # Less than 10% missing
                valid_features.append(col)
        
        self.feature_columns = valid_features
        # Forward fill then backward fill for missing values
        result = data[valid_features].copy()
        result = result.ffill().bfill()  # Modern pandas approach
        return result
    
    def train_models(self, X, y):
        """Train multiple models with hyperparameter tuning"""
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize models
        models_config = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {},
                'needs_scaling': True
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'needs_scaling': False
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                },
                'needs_scaling': False
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        for name, config in models_config.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            
            # Scale features if needed
            if config['needs_scaling']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[name] = scaler
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Hyperparameter tuning
            if config['params']:
                print("Performing hyperparameter tuning...")
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_model, y_train)
                best_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                best_model = config['model']
                best_model.fit(X_train_model, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                best_model, X_train_model, y_train, 
                cv=tscv, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Predictions
            predictions = best_model.predict(X_test_model)
            
            # Metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Store results
            self.models[name] = best_model
            self.results[name] = {
                'predictions': predictions,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'test_set': (X_test_model, y_test)
            }
            
            print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
            print(f"Test RMSE: {rmse:.2f}")
            print(f"Test MAE: {mae:.2f}")
            print(f"Test R²: {r2:.3f}")
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(f"\nTop 10 Features for {name}:")
                print(feature_importance.head(10))
    
    def create_ensemble_prediction(self, X):
        """Create ensemble predictions from all models"""
        predictions = {}
        weights = {}
        
        for name, model in self.models.items():
            # Prepare input
            if name in self.scalers:
                X_scaled = self.scalers[name].transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            predictions[name] = pred
            # Weight by R² score
            weights[name] = max(0, self.results[name]['r2'])
        
        # Weighted average
        total_weight = sum(weights.values())
        if total_weight > 0:
            ensemble_pred = sum(pred * weights[name] / total_weight 
                              for name, pred in predictions.items())
        else:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred, predictions

def create_advanced_visualizations(data, results):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(20, 16))
    
    # Temperature trend with rolling averages
    ax1 = plt.subplot(3, 4, 1)
    plt.plot(data['date'], data['temperature'], alpha=0.3, linewidth=0.5, label='Daily')
    plt.plot(data['date'], data['temperature'].rolling(30).mean(), 
             linewidth=2, label='30-day average')
    plt.title('Temperature Trends with Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Advanced correlation heatmap
    ax2 = plt.subplot(3, 4, 2)
    corr_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                'precipitation', 'heat_index', 'comfort_index']
    correlation_matrix = data[corr_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', square=True, ax=ax2)
    plt.title('Advanced Weather Correlations')
    
    # Model performance comparison
    ax3 = plt.subplot(3, 4, 3)
    models = list(results.keys())
    rmse_scores = [results[model]['rmse'] for model in models]
    r2_scores = [results[model]['r2'] for model in models]
    
    x = np.arange(len(models))
    ax3.bar(x - 0.2, rmse_scores, 0.4, label='RMSE', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + 0.2, r2_scores, 0.4, label='R²', alpha=0.7, color='orange')
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('RMSE')
    ax3_twin.set_ylabel('R² Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    plt.title('Model Performance Comparison')
    
    # Seasonal patterns
    ax4 = plt.subplot(3, 4, 4)
    seasonal_stats = data.groupby('season')['temperature'].agg(['mean', 'std'])
    seasons = seasonal_stats.index
    means = seasonal_stats['mean']
    stds = seasonal_stats['std']
    
    plt.bar(seasons, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    plt.title('Seasonal Temperature Patterns')
    plt.ylabel('Temperature (°F)')
    plt.xticks(rotation=45)
    
    # Best model predictions scatter
    ax5 = plt.subplot(3, 4, 5)
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    y_test = results[best_model]['test_set'][1]
    predictions = results[best_model]['predictions']
    
    plt.scatter(y_test, predictions, alpha=0.6, s=20)
    min_temp, max_temp = y_test.min(), y_test.max()
    plt.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8)
    plt.xlabel('Actual Temperature (°F)')
    plt.ylabel('Predicted Temperature (°F)')
    plt.title(f'Best Model: {best_model}\n(R² = {results[best_model]["r2"]:.3f})')
    
    # Weather indices over time
    ax6 = plt.subplot(3, 4, 6)
    recent_data = data.tail(365)  # Last year
    plt.plot(recent_data['date'], recent_data['heat_index'], 
             label='Heat Index', alpha=0.7)
    plt.plot(recent_data['date'], recent_data['comfort_index'], 
             label='Comfort Index', alpha=0.7)
    plt.title('Weather Comfort Indices (Last Year)')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Monthly precipitation patterns
    ax7 = plt.subplot(3, 4, 7)
    monthly_precip = data.groupby('month')['precipitation'].agg(['mean', 'sum'])
    months = range(1, 13)
    plt.bar(months, monthly_precip['mean'], alpha=0.7, color='lightblue')
    plt.title('Average Monthly Precipitation')
    plt.xlabel('Month')
    plt.ylabel('Precipitation (inches)')
    plt.xticks(months)
    
    # Wind speed distribution by season
    ax8 = plt.subplot(3, 4, 8)
    for season in data['season'].unique():
        season_data = data[data['season'] == season]['wind_speed']
        plt.hist(season_data, alpha=0.7, label=season, bins=20)
    plt.title('Wind Speed Distribution by Season')
    plt.xlabel('Wind Speed (mph)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Cross-validation scores
    ax9 = plt.subplot(3, 4, 9)
    cv_means = [results[model]['cv_rmse_mean'] for model in models]
    cv_stds = [results[model]['cv_rmse_std'] for model in models]
    
    plt.bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    plt.title('Cross-Validation RMSE Scores')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # Feature importance (for tree-based models)
    ax10 = plt.subplot(3, 4, 10)
    best_tree_model = None
    for model_name in ['XGBoost', 'Random Forest']:
        if model_name in results:
            best_tree_model = model_name
            break
    
    if best_tree_model:
        model = results[best_tree_model]
        # This would need the actual model object to get feature importance
        plt.text(0.5, 0.5, f'Feature Importance\n({best_tree_model})', 
                ha='center', va='center', transform=ax10.transAxes)
    
    plt.title('Model Feature Importance')
    
    # Residuals analysis
    ax11 = plt.subplot(3, 4, 11)
    best_predictions = results[best_model]['predictions']
    residuals = y_test - best_predictions
    plt.scatter(best_predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Temperature (°F)')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Analysis - {best_model}')
    
    # Year-over-year temperature comparison
    ax12 = plt.subplot(3, 4, 12)
    for year in data['year'].unique():
        year_data = data[data['year'] == year]
        plt.plot(year_data['day_of_year'], year_data['temperature'], 
                alpha=0.7, label=str(year))
    plt.title('Year-over-Year Temperature Comparison')
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("🌤️  Advanced NYC Weather Prediction Project")
    print("=" * 60)
    
    # Generate enhanced data
    print("Generating enhanced weather data...")
    generator = WeatherDataGenerator()
    raw_data = generator.generate_data()
    
    # Feature engineering
    print("Creating advanced features...")
    engineer = AdvancedFeatureEngineer()
    weather_data = engineer.create_features(raw_data)
    
    print(f"Generated {len(weather_data)} days of enhanced weather data")
    print(f"Features created: {len(weather_data.columns)} total columns")
    
    # Prepare features
    predictor = AdvancedWeatherPredictor()
    X = predictor.prepare_features(weather_data)
    y = weather_data['temperature']
    
    print(f"Final feature matrix: {X.shape}")
    print(f"Selected features: {len(predictor.feature_columns)}")
    
    # Train models
    predictor.train_models(X, y)
    
    # Create visualizations
    print("\nCreating advanced visualizations...")
    create_advanced_visualizations(weather_data, predictor.results)
    
    # Best model summary
    best_model = max(predictor.results.keys(), 
                    key=lambda x: predictor.results[x]['r2'])
    best_result = predictor.results[best_model]
    
    print(f"\n🎉 Advanced Analysis Complete!")
    print(f"=" * 60)
    print(f"Best Model: {best_model}")
    print(f"R² Score: {best_result['r2']:.3f}")
    print(f"RMSE: {best_result['rmse']:.2f}°F")
    print(f"Cross-validation RMSE: {best_result['cv_rmse_mean']:.2f} (±{best_result['cv_rmse_std']:.2f})°F")

if __name__ == "__main__":
    main()
