import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])

# Fitting a linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate residuals
residuals = y - y_pred

# Plotting residuals
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Calculate and print Mean Absolute Error (MAE)
print(f'Mean Absolute Error (MAE): {np.mean(np.abs(residuals)):.2f}')
