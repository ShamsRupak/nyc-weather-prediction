import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create mock data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
temperatures = np.random.normal(loc=60, scale=10, size=len(dates)) + 10 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))

# DataFrame
weather_data = pd.DataFrame({
    'date': dates,
    'temperature': temperatures
})

# Visualize data
sns.lineplot(x='date', y='temperature', data=weather_data)
plt.title('Temperature Trend in NYC')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature Engineering
weather_data['day_of_year'] = weather_data['date'].dt.dayofyear

# Split data
X = weather_data[['day_of_year']]
y = weather_data['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot Predictions
plt.scatter(X_test, y_test, color='blue', label='Actual Temperature')
plt.scatter(X_test, predictions, color='red', label='Predicted Temperature')
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°F)')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.show()

