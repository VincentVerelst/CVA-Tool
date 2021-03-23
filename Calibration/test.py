import numpy as np 
import pandas as pd 
from scipy import interpolate
from scipy.stats import norm
from scipy.optimize import fsolve
import scipy.optimize

def f(a, b, x):
	return a * np.power(x,2) + b


new_f = lambda x: np.power(np.abs(f(1,-4,x)),2) #take square of function and minimize it = same as finding root

minf = scipy.optimize.fmin(new_f, -2)

print(minf[0])
