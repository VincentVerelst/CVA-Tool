import numpy as np 
import pandas as pd 
import datetime
import math
import matplotlib.pyplot as plt
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from RiskDriverSimulation import *

#############################################################################
########## User Defined Data ############
#############################################################################
#All deals
fixedlegs = np.array([]) #Include all fixed-floating swaps you want to include in the netting set
floatlegs = np.array([1]) #Include all fixed-fixed swaps you want to include in the netting set
fxforwarddeals = np.array([]) #Include all FX Forwards you want to include in the netting set
swaptiondeals = np.array([]) #Include all swaptions you want to include in the netting set


#############################################################################
########## READ IN ALL INFOMRATION ############
#############################################################################

####Pricing Information
fixedleginput = pd.read_excel(r'Input/Runfiles/Pricing/fixedlegs.xlsx', skiprows=2, index_col=0) #index_col=0 is essential. With this you can reference the deal with the number you assigned to it, instead of its index
floatleginput = pd.read_excel(r'Input/Runfiles/Pricing/floatlegs.xlsx', skiprows=2, index_col=0)

#Monte Carlo Information
mcinput = pd.read_excel(r'Input/Runfiles/RiskDriverSimulation/MCDetails.xlsx')
valuation_date = mcinput['Valuation Date'][0]
end_date = mcinput['End Date Netting Set'][0]
correlation = mcinput['Correlation'][0]
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
irinput = pd.read_excel(r'Input/Runfiles/RiskDriverSimulation/IRinput.xlsx')
irinput = irinput.dropna(1, how='all') #Drops all columns with NA values
irinput = irinput.drop(irinput.columns[0], axis=1) #Drops first column
iramount = irinput.count(1)[0] #Count the amount of currencies
final_discount_curve = pd.read_excel(r'Input/Curves/' + irinput['domestic'][4] + '.xlsx') #Curve with which final future_mtms are discounted with to today

#Read in FX
fxinput = pd.read_excel(r'Input/Runfiles/RiskDriverSimulation/FXinput.xlsx')
fxinput = fxinput.dropna(1) #Drops all columns with NA values
fxinput = fxinput.drop(fxinput.columns[0], axis=1) #Drops first column
fxamount = fxinput.count(1)[0] #Count the amount of currencies
if(fxamount != iramount - 1):
	print("Error: amount of FX rates must be one less than amount of interest rates.")
	exit()

#Read in inflation (which is a combo of rates and FX)
inflationinput = pd.read_excel(r'Input/Runfiles/RiskDriverSimulation/InflationInput.xlsx')
inflationinput = inflationinput.drop(inflationinput.columns[0], axis=1) #Drops first column
inflationinput = inflationinput.dropna(1) #Drops all columns with NA values
inflationamount = inflationinput.count(1)[0] #Count the amount of currencies

equityinput = pd.read_excel(r'Input/Runfiles/RiskDriverSimulation/EquityInput.xlsx')
equityinput = equityinput.drop(equityinput.columns[0], axis=1) #Drops first column
equityinput = equityinput.dropna(1) #Drops all columns with NA values
equityamount = equityinput.count(1)[0] #Count the amount of currencies

correlationmatrix = pd.read_excel(r'Input/Correlation/' + correlation +  '.xlsx', header=None,skiprows=1)
correlationmatrix = correlationmatrix.drop(correlationmatrix.columns[0], axis=1)
correlationmatrix = correlationmatrix.values #Convert to numpy array (matrix)

num_rows, num_cols = correlationmatrix.shape
total = iramount + fxamount + inflationamount + equityamount 
#Check if dimensions of correlation matrix are correct
if(num_rows != total or num_cols != total): 
	print("Error: enter correlation matrix with correct dimensions: (2n-1)x(2n-1), with n = amount of risk drivers")
	exit()

#############################################################################
########## Monte Carlo Simulation ############
#############################################################################

irdrivers, fxdrivers, inflationdrivers, equitydrivers = create_riskdrivers(irinput, fxinput, inflationinput, equityinput)


chol, rand_matrices = mc_simulate_hwbs(irdrivers, fxdrivers, inflationdrivers, equitydrivers, correlationmatrix, timegrid, simulation_amount)

shortrates, fxrates = ir_fx_simulate(timegrid, simulation_amount, irdrivers, fxdrivers, rand_matrices, correlationmatrix)


# avgdomrate = np.mean(shortrates[0].get_simulated_rates(), axis=0)
# # avgforrate = np.mean(shortrates[1].get_simulated_rates(), axis=0)
# # avgfxrate = np.mean(fxrates[0].get_simulated_rates(), axis=0)

# # plt.plot(timegrid, avgdomrate)
# # plt.show()


# #############################################################################
# ########## Pricing ############
# #############################################################################

net_future_mtm = np.zeros((simulation_amount, len(timegrid)))

net_future_mtm = fixedpricing(fixedlegs, net_future_mtm, fixedleginput, timegrid, shortrates, fxrates, simulation_amount, irinput)

net_future_mtm = floatpricing(floatlegs, net_future_mtm, floatleginput, timegrid, shortrates, fxrates, simulation_amount, irinput)




# #stochastic discounting to today
net_discounted_mtm = stochastic_discount(net_future_mtm, shortrates['domestic'], timegrid, final_discount_curve)

print(np.mean(net_future_mtm, axis=0))
print(np.mean(net_discounted_mtm, axis=0))




