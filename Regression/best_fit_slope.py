#Programming Best-Fit Slope
#m = (xbar*ybar - (x*y)bar)/((xbar)^2 - (x^2)bar)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = [1,2,3,4,5,6]
ys = [5,4,6,5,6,7]

#change them to numpy arrays
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#plt.scatter(xs, ys)
#plt.show()

def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs*ys))/((mean(xs)**2) - (mean((xs)**2)))
    return m

m = best_fit_slope(xs,ys)
print(m)
