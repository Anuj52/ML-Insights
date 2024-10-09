import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import shap
import matplotlib.pyplot as plt

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X = pd.DataFrame(X, columns=housing.feature_names)

# Optionally, use a smaller subset of data to prevent memory issues
# X_sample = X.sample(1000, random_state=1)
# y_sample = y[:1000]

# Fit a decision tree regressor (using full data, or use X_sample and y_sample if memory issues)
model = DecisionTreeRegressor()
model.fit(X, y)

# Create SHAP explainer using TreeExplainer (optimized for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot SHAP values (ensure plt.show() is used to display the plot in some environments)
shap.summary_plot(shap_values, X)
plt.show()  # Ensures the plot is displayed in non-interactive environments
