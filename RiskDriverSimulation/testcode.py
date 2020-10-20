import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
from QuantLib import *
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
#from generalfunctions import *


x = np.array([1,2,3])

for i in range(0, len(x)):
	print(x[i])