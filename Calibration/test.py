import numpy as np 
import pandas as pd 
from scipy import interpolate
from scipy.stats import norm
from scipy.optimize import fsolve
import scipy.optimize

x = np.arange(1, 3+1, 1)

y = np.ones([3,3])
z = np.inf

print(z>1e20)
print(x*y)

z = np.zeros(1)

print(hasattr(z, "__len__"))