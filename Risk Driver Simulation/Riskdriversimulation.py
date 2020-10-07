import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from riskdrivers import *
from generalfunctions import *

#############################################################################
########## READ IN ALL INFOMRATION ############
#############################################################################

#Monte Carlo Information
mcinput = pd.read_excel(r'Runfiles/MCDetails.xlsx')
valuation_date = mcinput['Valuation Date'][0]
end_date = mcinput['End Date Netting Set'][0]
timesteps = mcinput['Timesteps'][0]
simulation_amount = mcinput['SimAmount'][0]
switcher = {
	"Yearly": 1,
	"Quarterly": 0.25,
	"Monthly": 1/12,
	"Weekly": 1/52,
	"Daily": 1/252
}
maturity_yearfrac = yf.yearfrac(valuation_date, end_date)

timegrid = np.arange(0, yf.yearfrac(valuation_date, end_date), step=switcher[timesteps])
if maturity_yearfrac not in timegrid:
	timegrid = np.append(timegrid, maturity_yearfrac)

#Read in rates
irinput = pd.read_excel(r'Runfiles/IRinput.xlsx')
irinput = irinput.dropna(1) #Drops all columns with NA values
irinput = irinput.drop(irinput.columns[0], axis=1) #Drops first column
currencyamount = irinput.count(1)[0] #Count the amount of currencies


#Read in FX
fxinput = pd.read_excel(r'Runfiles/FXinput.xlsx')
fxinput = fxinput.dropna(1) #Drops all columns with NA values
fxinput = fxinput.drop(fxinput.columns[0], axis=1) #Drops first column

#Read in inflation (which is a combo of rates and FX)
inflationinput = pd.read_excel(r'Runfiles/InflationInput.xlsx')
inflationinput = inflationinput.drop(inflationinput.columns[0], axis=1) #Drops first column
inflationinput = inflationinput.dropna(1) #Drops all columns with NA values
inflationamount = inflationinput.count(1)[0] #Count the amount of currencies

equityinput = pd.read_excel(r'Runfiles/EquityInput.xlsx')
equityinput = equityinput.drop(equityinput.columns[0], axis=1) #Drops first column
equityinput = equityinput.dropna(1) #Drops all columns with NA values
equityamount = equityinput.count(1)[0] #Count the amount of currencies

correlationmatrix = pd.read_excel(r'Input/Correlation/correlationmatrix.xlsx', header=None,skiprows=1)
correlationmatrix = correlationmatrix.drop(correlationmatrix.columns[0], axis=1)
correlationmatrix = correlationmatrix.values #Convert to numpy array (matrix)

num_rows, num_cols = correlationmatrix.shape
total = currencyamount + inflationamount + equityamount
#Check if dimensions of correlation matrix are correct
if(num_rows != 2*total-1 or num_cols != 2*total-1):
	print("Error: enter correlation matrix with correct dimensions: (2n-1)x(2n-1), with n = amount of risk drivers")
	exit()


irdrivers, fxdrivers, inflationdrivers, equitydrivers = create_riskdrivers(irinput, fxinput, inflationinput, equityinput)

#hwbs = HullWhiteBlackScholes(irinput, fxinput, inflationinput, equityinput, correlationmatrix, simulation_amount, timegrid)


print(irdrivers[0].get_meanreversion())
#Choleskycorr = np.linalg.cholesky(Correlationmatrix)
