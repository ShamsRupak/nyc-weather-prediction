"""
NYC Weather Prediction Project
==============================
A comprehensive machine learning project to predict weather trends in New York City
using scikit-learn with mock data that simulates realistic weather patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_mock_weather_data(start_date='2020-01-01', end_date='2024-12-31'):
    """
    Generate realistic mock weather data for NYC with seasonal patterns
    """
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic temperature data with seasonal variation
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base temperature with seasonal pattern (warmer in summer, colder in winter)
    base_temp = 50 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Add random noise
    temperature = base_temp + np.random.normal(0, 8, n_days)
    
    # Generate humidity (inversely related to temperature with noise)
    humidity = 70 - 0.3 * (temperature - 50) + np.random.normal(0, 10, n_days)
    humidity = np.clip(humidity, 20, 95)  # Keep realistic range
    
    # Generate atmospheric pressure (with seasonal variation)
    pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, n_days)
    
    # Generate wind speed (higher in winter)
    wind_speed = 10 + 5 * np.sin(2 * np.pi * (day_of_year + 180) / 365) + np.random.exponential(3, n_days)
    wind_speed = np.clip(wind_speed, 0, 30)
    
    # Generate precipitation (random with seasonal bias)
    precipitation = np.random.exponential(0.1, n_days)
    # More rain in spring/summer
    seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    precipitation *= seasonal_factor
    precipitation = np.clip(precipitation, 0, 5)  # Max 5 inches per day
    
    # Create DataFrame
    weather_data = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'precipitation': precipitation
    })
    
    # Add time-based features
    weather_data['year'] = weather_data['date'].dt.year
    weather_data['month'] = weather_data['date'].dt.month
    weather_data['day_of_year'] = weather_data['date'].dt.dayofyear
    weather_data['day_of_week'] = weather_data['date'].dt.dayofweek
    weather_data['season'] = weather_data['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    return weather_data

def create_visualizations(data):
    """
    Create comprehensive visualizations of the weather data
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NYC Weather Data Analysis', fontsize=16, fontweight='bold')
    
    # Temperature trend over time
    axes[0, 0].plot(data['date'], data['temperature'], alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Temperature Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Temperature (°F)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Temperature distribution by season
    sns.boxplot(data=data, x='season', y='temperature', ax=axes[0, 1])
    axes[0, 1].set_title('Temperature Distribution by Season')
    axes[0, 1].set_ylabel('Temperature (°F)')
    
    # Correlation heatmap
    numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 2])
    axes[0, 2].set_title('Weather Variables Correlation')
    
    # Monthly temperature averages
    monthly_avg = data.groupby('month')['temperature'].mean()
    axes[1, 0].bar(monthly_avg.index, monthly_avg.values, color='skyblue')
    axes[1, 0].set_title('Average Temperature by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Temperature (°F)')
    axes[1, 0].set_xticks(range(1, 13))
    
    # Humidity vs Temperature scatter
    axes[1, 1].scatter(data['temperature'], data['humidity'], alpha=0.5, s=1)
    axes[1, 1].set_title('Humidity vs Temperature')
    axes[1, 1].set_xlabel('Temperature (°F)')
    axes[1, 1].set_ylabel('Humidity (%)')
    
    # Precipitation distribution
    axes[1, 2].hist(data['precipitation'], bins=50, alpha=0.7, color='lightblue')
    axes[1, 2].set_title('Precipitation Distribution')
    axes[1, 2].set_xlabel('Precipitation (inches)')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def prepare_features(data):
    """
    Prepare features for machine learning models
    """
    # Select features for prediction
    feature_columns = [
        'day_of_year', 'month', 'day_of_week', 
        'humidity', 'pressure', 'wind_speed', 'precipitation'
    ]
    
    X = data[feature_columns].copy()
    y = data['temperature'].copy()
    
    return X, y

def train_and_evaluate_models(X, y):
    """
    Train multiple models and evaluate their performance
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        
        # Train model
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results[name] = {
            'model': model,
            'predictions': predictions,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance for Random Forest
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)
    
    return results, X_test, y_test

def create_prediction_visualizations(results, X_test, y_test):
    """
    Create visualizations comparing model predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red']
    
    for i, (name, result) in enumerate(results.items()):
        predictions = result['predictions']
        
        # Actual vs Predicted scatter plot
        axes[0].scatter(y_test, predictions, alpha=0.6, color=colors[i], 
                       label=f'{name} (R² = {result["r2"]:.3f})', s=20)
    
    # Perfect prediction line
    min_temp, max_temp = y_test.min(), y_test.max()
    axes[0].plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.8, linewidth=2)
    axes[0].set_xlabel('Actual Temperature (°F)')
    axes[0].set_ylabel('Predicted Temperature (°F)')
    axes[0].set_title('Actual vs Predicted Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Model performance metrics comparison
    metrics = ['RMSE', 'MAE', 'R²']
    linear_metrics = [results['Linear Regression']['rmse'], 
                     results['Linear Regression']['mae'], 
                     results['Linear Regression']['r2']]
    rf_metrics = [results['Random Forest']['rmse'], 
                 results['Random Forest']['mae'], 
                 results['Random Forest']['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, linear_metrics, width, label='Linear Regression', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, rf_metrics, width, label='Random Forest', color='red', alpha=0.7)
    
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Values')
    axes[1].set_title('Model Performance Metrics')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (linear_val, rf_val) in enumerate(zip(linear_metrics, rf_metrics)):
        axes[1].text(i - width/2, linear_val + 0.01, f'{linear_val:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        axes[1].text(i + width/2, rf_val + 0.01, f'{rf_val:.2f}', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def predict_future_weather(model, scaler, data, days_ahead=30):
    """
    Predict weather for the next N days
    """
    print(f"\n{'='*50}")
    print(f"Predicting weather for the next {days_ahead} days...")
    print(f"{'='*50}")
    
    # Get the last date in the dataset
    last_date = data['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=days_ahead, freq='D')
    
    # Create future features (simplified - using seasonal patterns)
    future_data = []
    for date in future_dates:
        day_of_year = date.timetuple().tm_yday
        month = date.month
        day_of_week = date.dayofweek
        
        # Use average values for weather features (in real scenario, you'd use weather forecasts)
        avg_humidity = data['humidity'].mean()
        avg_pressure = data['pressure'].mean()
        avg_wind_speed = data['wind_speed'].mean()
        avg_precipitation = data['precipitation'].mean()
        
        future_data.append([day_of_year, month, day_of_week, 
                          avg_humidity, avg_pressure, avg_wind_speed, avg_precipitation])
    
    future_features = pd.DataFrame(future_data, 
                                  columns=['day_of_year', 'month', 'day_of_week', 
                                          'humidity', 'pressure', 'wind_speed', 'precipitation'])
    
    # Scale features and predict
    future_features_scaled = scaler.transform(future_features)
    future_predictions = model.predict(future_features_scaled)
    
    # Create results DataFrame
    future_results = pd.DataFrame({
        'date': future_dates,
        'predicted_temperature': future_predictions
    })
    
    print("Future Weather Predictions:")
    print(future_results.head(10))
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    
    # Plot last 90 days of actual data
    recent_data = data.tail(90)
    plt.plot(recent_data['date'], recent_data['temperature'], 
            label='Historical Temperature', color='blue', linewidth=2)
    
    # Plot future predictions
    plt.plot(future_results['date'], future_results['predicted_temperature'], 
            label='Predicted Temperature', color='red', linewidth=2, linestyle='--')
    
    plt.axvline(x=last_date, color='green', linestyle=':', linewidth=2, 
               label='Prediction Start')
    
    plt.title(f'NYC Temperature: Historical Data and {days_ahead}-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return future_results

def main():
    """
    Main function to run the complete weather prediction pipeline
    """
    print("🌤️  NYC Weather Prediction Project")
    print("=====================================")
    print("Generating mock weather data...")
    
    # Generate mock data
    weather_data = generate_mock_weather_data()
    
    print(f"Generated {len(weather_data)} days of weather data")
    print(f"Date range: {weather_data['date'].min()} to {weather_data['date'].max()}")
    print("\nDataset Info:")
    print(weather_data.info())
    print("\nFirst few rows:")
    print(weather_data.head())
    
    # Create visualizations
    print("\nCreating data visualizations...")
    create_visualizations(weather_data)
    
    # Prepare features
    print("Preparing features for machine learning...")
    X, y = prepare_features(weather_data)
    
    print(f"Features: {list(X.columns)}")
    print(f"Target: Temperature")
    print(f"Dataset shape: {X.shape}")
    
    # Train and evaluate models
    results, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Create prediction visualizations
    create_prediction_visualizations(results, X_test, y_test)
    
    # Future predictions using the best model (Random Forest)
    best_model = results['Random Forest']['model']
    
    # Create scaler for future predictions
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Actually, Random Forest doesn't need scaling, so we'll use Linear Regression for future predictions
    linear_model = results['Linear Regression']['model']
    future_predictions = predict_future_weather(linear_model, scaler, weather_data)
    
    print("\n🎉 Weather prediction analysis complete!")
    print("Summary of best model performance:")
    best_result = max(results.values(), key=lambda x: x['r2'])
    print(f"Best R² Score: {best_result['r2']:.3f}")
    print(f"Best RMSE: {best_result['rmse']:.2f}°F")

if __name__ == "__main__":
    main()
