import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import shap
import matplotlib.pyplot as plt

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X = pd.DataFrame(X, columns=housing.feature_names)

# Fit a decision tree regressor
model = DecisionTreeRegressor()
model.fit(X, y)

# Create SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot SHAP values
shap.summary_plot(shap_values, X)


