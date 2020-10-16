import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt


#Read in inflation (which is a combo of rates and FX)
x = 3

for i in range(0,3):
	print(x)