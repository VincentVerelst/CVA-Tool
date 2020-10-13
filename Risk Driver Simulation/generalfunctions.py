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


def ir_fx_simulate(timegrid, simulation_amount, irdrivers, fxdrivers, random_matrices, correlationmatrix):
	simulated_shortrates = []
	simulated_fxrates = []

	deltaT = np.diff(timegrid)
	#deltaT = timegrid[2] - timegrid[1] #determine time intervals of timegrid
	n_irdrivers = len(irdrivers)
	n_fxdrivers = len(fxdrivers)

	#Calculate deterministic beta factors so that r(t) = x(t) + beta(t), with x(t) ornstein uhlenbeck process
	betas = []
	for i in range(0,len(irdrivers)):
		beta = irdrivers[i].get_beta(timegrid)
		betas.append(beta)
		
	#Simulate the short rates
	short_rates = []
	#Simulate domestic short rate (OU)
	domestic_ou = np.zeros((simulation_amount, len(timegrid)))
	domestic_short_rates = np.zeros((simulation_amount, len(timegrid)))
	domestic_short_rates[:,0] = betas[0][0]
	for i in range(0, len(timegrid) - 1):
		#Exact solution scheme for x(t)
		domestic_ou[:,i+1] = domestic_ou[:,i] * np.exp(- irdrivers[0].get_meanreversion() * deltaT[i]) + irdrivers[0].get_volatility(timegrid[i+1]) * random_matrices[0][:,i] * np.sqrt((1 - np.exp(-2 * irdrivers[0].get_meanreversion() * deltaT[i]))/(2 * irdrivers[0].get_meanreversion()))
		domestic_short_rates[:, i+1] = domestic_ou[:,i+1] + betas[0][i+1]

	short_rates.append(domestic_short_rates)
	#Simulate foreign short rate (OU)
	for j in range(1, n_irdrivers):
		foreign_ou = np.zeros((simulation_amount, len(timegrid)))
		foreign_short_rates = np.zeros((simulation_amount, len(timegrid)))
		foreign_short_rates[:,0] = betas[j][0]
		for i in range(0, len(timegrid) - 1):
			#Exact solution scheme for x(t)
			foreign_ou[:,i+1] = foreign_ou[:,i] * np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) + (irdrivers[j].get_volatility(timegrid[i+1]) * fxdrivers[j-1].get_volatility(timegrid[i+1]) * correlationmatrix[j][n_irdrivers + j - 1]/irdrivers[j].get_meanreversion()) * (np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) - 1) + irdrivers[j].get_volatility(timegrid[i+1]) * random_matrices[j][:,i] * np.sqrt((1 - np.exp(-2 * irdrivers[j].get_meanreversion() * deltaT[i]))/(2 * irdrivers[j].get_meanreversion()))
			foreign_short_rates[:, i+1] = foreign_ou[:,i+1] + betas[j][i+1]
		short_rates.append(foreign_short_rates)
	
	
	#simulate the spot FX rates
	fxspot_rates = []
	for j in range(0, n_fxdrivers):
		spotfx = np.zeros((simulation_amount, len(timegrid)))
		spotfx[:,0] = fxdrivers[j].get_spotfx() 
		for i in range(0, len(timegrid) - 1):
			spotfx[:,i+1] = spotfx[:,i] + spotfx[:,i] * ( (short_rates[0][:,i+1] - short_rates[j+1][:,i+1]) * deltaT[i] + fxdrivers[j].get_volatility(timegrid[i+1]) * np.sqrt(deltaT[i]) * random_matrices[n_irdrivers + j][:,i] )

		fxspot_rates.append(spotfx)
	
	return(short_rates, fxspot_rates)



def mc_simulate_hwbs(irdrivers, fxdrivers, inflationdrivers, equitydrivers, correlationmatrix, timegrid, simulation_amount):
	#Count riskdrivers
	n_irdrivers = len(irdrivers)
	n_fxdrivers = len(fxdrivers)
	n_inflationdrivers = len(inflationdrivers)
	n_equitydrivers = len(equitydrivers)
	n_totaldrivers = n_irdrivers + n_fxdrivers + n_inflationdrivers + n_equitydrivers

	#Generate antithetic correlated random paths using cholesky decomposition
	cholesky_correlationmatrix = np.linalg.cholesky(correlationmatrix)
	#Generate random independent matrices with antithetic paths
	random_matrices = []
	for i in range(0, n_totaldrivers):
		rand_matrix = np.random.randn(int(simulation_amount / 2),  len(timegrid))
		rand_matrix = np.concatenate((rand_matrix, -rand_matrix)) #antithetic paths
		random_matrices.append(rand_matrix)

	#Correlate the independent matrices with cholesky decomp
	correlated_random_matrices = []
	for i in range(0, n_totaldrivers):
		rand_matrix = np.zeros((simulation_amount, len(timegrid)))
		for j in range(0, n_totaldrivers):
			rand_matrix += random_matrices[j] * cholesky_correlationmatrix[i][j]
		correlated_random_matrices.append(rand_matrix)

	return(cholesky_correlationmatrix, correlated_random_matrices)

