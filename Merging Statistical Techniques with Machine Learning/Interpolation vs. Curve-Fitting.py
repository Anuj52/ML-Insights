import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial

# Generate synthetic data
x = np.linspace(0, 10, 10)
y = np.sin(x) + np.random.normal(0, 0.1, 10)

# Interpolation
interp_func = interp1d(x, y, kind='linear')

# Curve fitting (2nd degree polynomial)
p = Polynomial.fit(x, y, 2)

# Plotting
x_new = np.linspace(0, 10, 100)
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_new, interp_func(x_new), color='green', label='Interpolation')
plt.plot(x_new, p(x_new), color='red', label='Curve Fitting (2nd Degree Polynomial)')
plt.title('Interpolation vs Curve Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
