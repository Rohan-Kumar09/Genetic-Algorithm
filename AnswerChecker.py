import numpy as np
import math

def my_function(x, y):
    return ((1-x)**2 * math.e**(-x**2 - (y+1)**2)) - (x - x**3 - y**3) * math.e**(-x**2 - y**2)
    # term1 = (1 - x**2) * np.exp(-(x**2 + y**2))
    # term2 = (x**3 - y) * np.exp(-((x-1)**2 + (y-1)**2))
    # return term1 + term2

# Define the range
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = my_function(X, Y)

# Find global max and min
global_max = np.max(Z)
# global_min = np.min(Z)

print(global_max)
