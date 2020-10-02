import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
import hullwhiteblackscholes

#Read in inflation (which is a combo of rates and FX)
inflationinput = pd.read_excel(r'Runfiles/InflationInput.xlsx')
inflationinput = inflationinput.drop(inflationinput.columns[0], axis=1) #Drops first column
inflationinput = inflationinput.dropna(1) #Drops all columns with NA values
inflationamount = inflationinput.count(1)[0] #Count the amount of currencies

print(inflationinput.count(axis=1)[0])