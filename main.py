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
fixedlegs = np.arange(1,18) #Include all fixed-floating swaps you want to include in the netting set
floatlegs = np.arange(1,16) #Include all fixed-fixed swaps you want to include in the netting set
cmslegs = np.array([1,2])#np.array([1,2])
fxforwarddeals = np.array([]) #Include all FX Forwards you want to include in the netting set
swaptiondeals = np.array([]) #Include all swaptions you want to include in the netting set


#############################################################################
########## READ IN ALL INFOMRATION ############
#############################################################################

####Pricing Information
dateparse = lambda x: x if isinstance(x, datetime.date) else datetime.datetime.strptime(x, '%d/%m/%Y') #if data is already datetime, don't change it, else convert it to datetime object
fixedleginput = pd.read_excel(r'Input/Runfiles/Pricing/fixedlegs.xlsx', skiprows=2, index_col=0, parse_dates=['ValDate','StartDate', 'EndDate'], date_parser=dateparse) #index_col=0 is essential. With this you can reference the deal with the number you assigned to it, instead of its index
floatleginput = pd.read_excel(r'Input/Runfiles/Pricing/floatlegs.xlsx', skiprows=2, index_col=0, parse_dates=['ValDate','StartDate', 'EndDate'], date_parser=dateparse)
cmsleginput = pd.read_excel(r'Input/Runfiles/Pricing/cmslegs.xlsx', skiprows=2, index_col=0, parse_dates=['ValDate','StartDate', 'EndDate'], date_parser=dateparse)


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
total = iramount + fxamount + 2*inflationamount + equityamount #2* inflation amount because every inflation drivers contains two extra risk drivers: the real rate and the inflation index 
#Check if dimensions of correlation matrix are correct
if(num_rows != total or num_cols != total): 
	print("Error: enter correlation matrix with correct dimensions: N x N, with N = amount of risk drivers")
	exit()

#############################################################################
########## Monte Carlo Simulation ############
#############################################################################

irdrivers, fxdrivers, inflationdrivers, equitydrivers = create_riskdrivers(irinput, fxinput, inflationinput, equityinput)


chol, rand_matrices = mc_simulate_hwbs(irdrivers, fxdrivers, inflationdrivers, equitydrivers, correlationmatrix, timegrid, simulation_amount)

print(len(rand_matrices))

shortrates, fxrates, inflationrates = ir_fx_simulate(timegrid, simulation_amount, irdrivers, fxdrivers, inflationdrivers, rand_matrices, correlationmatrix)



# # avgdomrate = np.mean(shortrates[0].get_simulated_rates(), axis=0)
# # # avgforrate = np.mean(shortrates[1].get_simulated_rates(), axis=0)
# # # avgfxrate = np.mean(fxrates[0].get_simulated_rates(), axis=0)

# # # plt.plot(timegrid, avgdomrate)
# # # plt.show()



# # # #############################################################################
# # # ########## Pricing ############
# # # #############################################################################

# net_future_mtm = np.zeros((simulation_amount, len(timegrid)))

# net_future_mtm = fixedpricing(fixedlegs, net_future_mtm, fixedleginput, timegrid, shortrates, fxrates, simulation_amount)

# net_future_mtm = floatpricing(floatlegs, net_future_mtm, floatleginput, timegrid, shortrates, fxrates, simulation_amount)

# net_future_mtm = cmslegpricing(cmslegs, net_future_mtm, cmsleginput, timegrid, shortrates, fxrates, simulation_amount)


# # #stochastic discounting to today
# net_discounted_mtm = stochastic_discount(net_future_mtm, shortrates['domestic'], timegrid, final_discount_curve)


# # # #############################################################################
# # # ########## Exposure Calculation + Writing to Excel ############
# # # #############################################################################

# #Expected Exposure
# EE = np.mean(net_discounted_mtm, axis=0)

# #Expected Positive Exposure
# PE = net_discounted_mtm.copy()
# PE[PE < 0] = 0
# EPE = np.mean(PE, axis=0)

# #Expected Negative Exposure
# NE = net_discounted_mtm.copy()
# NE[NE > 0] = 0
# ENE = np.mean(NE, axis=0)

# #Create a dataframe with all data
# output = pd.DataFrame({"Tenor [Y]": timegrid, "EE": EE, "EPE": EPE, "ENE":ENE} )

# #Write to Excel
# output.to_excel("Output/exposures.xlsx")


# print(EE[0])


