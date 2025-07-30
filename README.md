# NYC Weather Prediction Project 🌤️

A comprehensive machine learning project that predicts weather trends in New York City using scikit-learn and realistic mock weather data.

## 📋 Project Overview

This project demonstrates end-to-end machine learning pipeline for weather prediction, including:
- **Data Generation**: Creates realistic mock weather data with seasonal patterns
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Compares Linear Regression and Random Forest algorithms
- **Future Predictions**: Forecasts weather trends for upcoming days
- **Performance Evaluation**: Detailed metrics and model comparison

## 🚀 Features

- **5-year historical data simulation** (2020-2024) with 1,827 data points
- **Multiple weather variables**: Temperature, humidity, pressure, wind speed, precipitation
- **Seasonal patterns**: Realistic seasonal variations in all weather parameters
- **Time-based features**: Day of year, month, day of week, season
- **Advanced visualizations**: 6 different charts showing data patterns and correlations
- **Model comparison**: Linear Regression vs Random Forest with detailed metrics
- **Future forecasting**: 30-day weather predictions

## 📊 Results Summary

### Best Model Performance (Random Forest):
- **R² Score**: 0.814 (81.4% variance explained)
- **RMSE**: 8.45°F
- **MAE**: 6.67°F

### Feature Importance:
1. **Day of Year**: 83.2% (Most important - captures seasonal patterns)
2. **Humidity**: 4.1%
3. **Month**: 3.0%
4. **Precipitation**: 2.8%
5. **Pressure**: 2.8%
6. **Wind Speed**: 2.8%
7. **Day of Week**: 1.3%

## 🛠️ Technologies Used

- **Python 3.13.5**
- **scikit-learn 1.7.1**: Machine learning algorithms
- **pandas 2.3.1**: Data manipulation and analysis
- **numpy 2.3.2**: Numerical computing
- **matplotlib 3.10.3**: Plotting and visualization
- **seaborn 0.13.2**: Statistical data visualization

## 📁 Project Structure

```
nyc_weather_prediction/
├── venv/                           # Virtual environment
├── weather_prediction.py          # Simple version
├── enhanced_weather_prediction.py # Comprehensive version
└── README.md                      # This file
```

## 🏃‍♂️ How to Run

### Prerequisites
- Python 3.13+ installed
- Virtual environment set up

### Installation & Execution

1. **Navigate to project directory**:
   ```bash
   cd nyc_weather_prediction
   ```

2. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Run the enhanced version**:
   ```bash
   python enhanced_weather_prediction.py
   ```

### Expected Output
The script will display:
- Dataset information and first few rows
- 6 comprehensive visualizations showing weather patterns
- Model training progress with detailed metrics
- Feature importance rankings
- Model performance comparison charts
- 30-day future weather predictions
- Summary of best model performance

## 📈 Visualizations Generated

1. **Temperature Over Time**: 5-year temperature trend
2. **Seasonal Temperature Distribution**: Box plots by season
3. **Correlation Heatmap**: Relationships between weather variables
4. **Monthly Temperature Averages**: Bar chart of average temps by month
5. **Humidity vs Temperature**: Scatter plot showing inverse relationship
6. **Precipitation Distribution**: Histogram of rainfall patterns
7. **Model Performance Comparison**: Actual vs predicted scatter plots
8. **Metrics Comparison**: Bar charts comparing RMSE, MAE, and R²
9. **Future Predictions**: 30-day forecast with historical context

## 🔬 Data Features

### Generated Weather Variables:
- **Temperature**: Seasonal pattern with realistic NYC ranges (20-85°F)
- **Humidity**: Inversely correlated with temperature (20-95%)
- **Pressure**: Atmospheric pressure with seasonal variation (~1013 hPa)
- **Wind Speed**: Higher in winter, exponential distribution (0-30 mph)
- **Precipitation**: Seasonal bias toward spring/summer (0-5 inches)

### Engineered Features:
- **Day of Year**: 1-365/366 (captures seasonal cycles)
- **Month**: 1-12 (monthly patterns)
- **Day of Week**: 0-6 (weekly patterns, if any)
- **Season**: Winter, Spring, Summer, Fall

## 🎯 Model Comparison

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | 16.34°F | 13.49°F | 0.303 |
| **Random Forest** | **8.45°F** | **6.67°F** | **0.814** |

**Winner**: Random Forest significantly outperforms Linear Regression, capturing complex non-linear relationships in weather data.

## 🔮 Future Predictions

The model generates 30-day temperature forecasts using:
- Historical seasonal patterns
- Time-based features (day of year, month)
- Average weather conditions for unknown variables

Sample prediction output:
```
2025-01-01: 51.7°F
2025-01-02: 51.5°F
2025-01-03: 51.4°F
...
```

## 📚 Key Learning Outcomes

1. **Data Generation**: Creating realistic synthetic datasets with domain knowledge
2. **Feature Engineering**: Extracting meaningful time-based features
3. **Model Selection**: Comparing different algorithms for regression tasks
4. **Evaluation Metrics**: Understanding RMSE, MAE, and R² in context
5. **Visualization**: Creating comprehensive charts for data exploration
6. **Seasonal Modeling**: Capturing cyclical patterns in time series data

## 🔧 Potential Improvements

1. **Advanced Models**: Try XGBoost, LSTM, or Prophet for time series
2. **External Data**: Incorporate real weather APIs or historical data
3. **Feature Engineering**: Add moving averages, lag features, weather indices
4. **Hyperparameter Tuning**: Grid search for optimal model parameters
5. **Cross-Validation**: Time-series specific validation strategies
6. **Ensemble Methods**: Combine multiple models for better predictions

## 📖 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NYC Weather Patterns](https://www.weather.gov/okx/NYC_weather_stats)
- [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Time Series Feature Engineering](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

---

**Created by**: Weather Prediction ML Project  
**Date**: July 2025  
**Version**: 2.0  
**License**: MIT
