import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from scipy import interpolate
from riskdrivers import *

#Function that creates riskdrivers
def create_riskdrivers(irinput, fxinput, inflationinput, equityinput):
	#Define all the irdriver objects
	irdrivers = []
	for i in range(0, irinput.count(axis=1)[0]):
		temp_name = irinput[irinput.columns[i]][0]
		temp_yieldcurve = pd.read_excel(r'Input/Curves/' + irinput[irinput.columns[i]][1] + '.xlsx')
		temp_volatility = pd.read_excel(r'Input/Volatility/' + irinput[irinput.columns[i]][2] + '.xlsx')
		temp_mean_reversion = irinput[irinput.columns[i]][3]
		irdriver = RatesDriver(temp_name, temp_yieldcurve, temp_volatility, temp_mean_reversion)
		irdrivers.append(irdriver)

	fxdrivers = []
	for i in range(0, fxinput.count(axis=1)[0]):
		temp_name = fxinput[fxinput.columns[i]][0]
		temp_spotfx = fxinput[fxinput.columns[i]][1]
		temp_volatility = pd.read_excel(r'Input/Volatility/' + fxinput[fxinput.columns[i]][2] + '.xlsx')
		fxdriver = FXDriver(temp_name, temp_spotfx, temp_volatility)
		fxdrivers.append(fxdriver)

	inflationdrivers = []
	for i in range(0, inflationinput.count(axis=1)[0]):
		temp_name = inflationinput[inflationinput.columns[i]][0]
		temp_real_rate = pd.read_excel(r'Input/Curves/' + inflationinput[inflationinput.columns[i]][1] + '.xlsx')
		temp_nominal_rate = pd.read_excel(r'Input/Curves/' + inflationinput[inflationinput.columns[i]][2] + '.xlsx')
		temp_initial_index = inflationinput[inflationinput.columns[i]][3]
		temp_real_volatility = pd.read_excel(r'Input/Volatility/' + inflationinput[inflationinput.columns[i]][4] + '.xlsx')
		temp_nominal_volatility = pd.read_excel(r'Input/Volatility/' + inflationinput[inflationinput.columns[i]][5] + '.xlsx')
		temp_index_volatility = pd.read_excel(r'Input/Volatility/' + inflationinput[inflationinput.columns[i]][6] + '.xlsx')
		temp_real_mean_reversion = inflationinput[inflationinput.columns[i]][7]
		temp_nominal_mean_reversion = inflationinput[inflationinput.columns[i]][8]
		inflationdriver = InflationDriver(temp_name, temp_real_rate, temp_nominal_rate, temp_initial_index, temp_real_volatility, temp_nominal_volatility, temp_index_volatility, temp_real_mean_reversion, temp_nominal_mean_reversion)
		inflationdrivers.append(inflationdriver)

	#Define all the equitydriver objects
	equitydrivers = []
	for i in range(0, equityinput.count(axis=1)[0]):
		temp_name = equityinput[equityinput.columns[i]][0]
		temp_initial_index = equityinput[equityinput.columns[i]][1]
		temp_volatility = pd.read_excel(r'Input/Volatility/' + equityinput[equityinput.columns[i]][2] + '.xlsx')
		equitydriver = EquityDriver(temp_name, temp_initial_index, temp_volatility)
		equitydrivers.append(equitydriver)

	return(irdrivers, fxdrivers, inflationdrivers, equitydrivers)