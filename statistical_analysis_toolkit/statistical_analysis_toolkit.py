import numpy as np
from scipy import stats

# Sample dataset
data = [1, 2, 3, 4, 5, 6, 100]

# Mean
mean = np.mean(data)
print("Mean:", mean)

# Median
median = np.median(data)
print("Median:", median)

# Mode
mode_result = stats.mode(data, keepdims=True)  # Updated function with keepdims=True to maintain compatibility
print("Mode:", mode_result.mode[0])

# Variance
variance = np.var(data)
print("Variance:", variance)

# Standard Deviation
std_dev = np.std(data)
print("Standard Deviation:", std_dev)


import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data following a normal distribution
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# Plotting the distribution
sns.histplot(normal_data, kde=True)
plt.title('Normal Distribution')
plt.show()


# Skewness
skewness = stats.skew(data)
print("Skewness:", skewness)

# Kurtosis
kurtosis = stats.kurtosis(data)
print("Kurtosis:", kurtosis)

# Interquartile Range (IQR)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
print("IQR:", IQR)
