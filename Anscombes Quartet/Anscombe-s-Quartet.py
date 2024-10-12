import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Anscombe's Quartet dataset
anscombe = sns.load_dataset("anscombe")

# Print basic descriptive statistics
print("Descriptive Statistics for Anscombe's Quartet:")
print(anscombe.groupby("dataset").describe())

# Plot each dataset from Anscombe's Quartet
sns.set(style="ticks")

# Create a grid of scatter plots
g = sns.FacetGrid(anscombe, col="dataset", col_wrap=2, height=4)

# Scatter plot with linear regression fit
g.map(sns.scatterplot, "x", "y", s=100, color=".3")
g.map(sns.lineplot, "x", "y", color="red", errorbar=None)

# Add titles and labels
g.set_axis_labels("X", "Y")
g.set_titles("Dataset {col_name}")

# Display the plots
plt.tight_layout()
plt.show()
