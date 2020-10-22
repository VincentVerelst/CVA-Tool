import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from scipy import interpolate
from .riskdrivers import *

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
		#Create shortrates object containing all necessary info needed later in pricing
		domestic_short_rates_object = ShortRates(irdrivers[0].get_name(), domestic_short_rates, irdrivers[0].get_yieldcurve(), irdrivers[0].get_volatility_frame(), irdrivers[0].get_meanreversion(), timegrid, irdrivers[0].get_inst_fwd_rates(timegrid))

	short_rates.append(domestic_short_rates_object)
	#Simulate foreign short rate (OU)
	for j in range(1, n_irdrivers):
		foreign_ou = np.zeros((simulation_amount, len(timegrid)))
		foreign_short_rates = np.zeros((simulation_amount, len(timegrid)))
		foreign_short_rates[:,0] = betas[j][0]
		for i in range(0, len(timegrid) - 1):
			#Exact solution scheme for x(t)
			foreign_ou[:,i+1] = foreign_ou[:,i] * np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) + (irdrivers[j].get_volatility(timegrid[i+1]) * fxdrivers[j-1].get_volatility(timegrid[i+1]) * correlationmatrix[j][n_irdrivers + j - 1]/irdrivers[j].get_meanreversion()) * (np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) - 1) + irdrivers[j].get_volatility(timegrid[i+1]) * random_matrices[j][:,i] * np.sqrt((1 - np.exp(-2 * irdrivers[j].get_meanreversion() * deltaT[i]))/(2 * irdrivers[j].get_meanreversion()))
			foreign_short_rates[:, i+1] = foreign_ou[:,i+1] + betas[j][i+1]
		
		#Create shortrates object containing all necessary info needed later in pricing
		foreign_short_rates_object = ShortRates(irdrivers[j].get_name(), foreign_short_rates, irdrivers[j].get_yieldcurve(), irdrivers[j].get_volatility_frame(), irdrivers[j].get_meanreversion(), timegrid, irdrivers[j].get_inst_fwd_rates(timegrid))

		short_rates.append(foreign_short_rates_object)
	
	
	#simulate the spot FX rates
	fxspot_rates = []
	for j in range(0, n_fxdrivers):
		spotfx = np.zeros((simulation_amount, len(timegrid)))
		spotfx[:,0] = fxdrivers[j].get_spotfx() 
		for i in range(0, len(timegrid) - 1):
			spotfx[:,i+1] = spotfx[:,i] + spotfx[:,i] * ( (short_rates[0].get_simulated_rates()[:,i+1] - short_rates[j+1].get_simulated_rates()[:,i+1]) * deltaT[i] + fxdrivers[j].get_volatility(timegrid[i+1]) * np.sqrt(deltaT[i]) * random_matrices[n_irdrivers + j][:,i] )

		#Create fxrates object containing all necessary info needed later in pricing	
		spotfx_object = FXRates(fxdrivers[j].get_name(), spotfx, fxdrivers[j].get_spotfx(), fxdrivers[j].get_volatility_frame())
		fxspot_rates.append(spotfx_object)
	
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

def ql_to_datetime(d):
	return datetime.datetime(d.year(), d.month(), d.dayOfMonth())

def create_payment_times(frequency, startdate, enddate, valdate):
	cal = ql.UnitedKingdom() #We will consider the weekends and holidays of the UK as a proxy
	#Convert dates to quantlib date objects
	start_date = ql.Date(startdate.day, startdate.month, startdate.year)
	end_date = ql.Date(enddate.day, enddate.month, enddate.year)
	frequency = frequency*12 #Frequency in excel is expressed in years, but frequency in schedule must be in months
	paydates = ql.MakeSchedule(start_date, end_date, ql.Period(str(int(frequency)) + 'M'), calendar=cal, backwards=True, convention=ql.ModifiedFollowing)
	schedule = np.array([ql_to_datetime(d) for d in paydates]) #convert the quantlib dates back to datetime dates
	schedule = schedule[schedule > valdate] #remove all dates that are before the valuation date
	yearfrac = yf.yearfrac(valdate, schedule) #determine the yearfracs wrt the valdate
	yearfrac = np.array(yearfrac) #convert to numpy array
	return(yearfrac)

def include_yield_curve_basis(times, first_yield_curve, second_yield_curve, timegrid, n, stoch_discount_factors):
	#Both yield_curve inputs are pd. Dataframes with one colum 'TimeToZero' and the other one 'ZeroRate'
	#times is the times on which we want to have stoch dfs. times > timegrid[n] always
	#stoch_discount_factors is the matrix in which we want to include the deterministic basis between the two specified yield curves
	first_yield_fun = interpolate.interp1d(first_yield_curve['TimeToZero'], first_yield_curve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
	second_yield_fun = interpolate.interp1d(second_yield_curve['TimeToZero'], second_yield_curve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate

	first_rates = first_yield_fun(timegrid)
	second_rates = second_yield_fun(timegrid)
	diff = second_rates - first_rates 

	diff_fun = interpolate.interp1d(timegrid, diff, 'linear', fill_value='extrapolate') #linear interpolation of the difference between rates

	stoch_zero_rates = np.power(1 / stoch_discount_factors, 1 / (times - timegrid[n])) - 1 #convert original DF's to zero rates
	basisincluded_stoch_zero_rates = stoch_zero_rates + diff_fun(times - timegrid[n]) #add the deterministic basis (which is still time dependent)
	basisincluded_stoch_discount_factors = np.power(1 + basisincluded_stoch_zero_rates, -(times-timegrid[n]))
	return(basisincluded_stoch_discount_factors)

#Function to check whether input (from excel for example) can be converted to float or not
def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def fixedvalue(notional, freq, rate, discount_curve, timegrid, n, shortrates, futurepaytimes, nonotionalexchange= None):
	sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(futurepaytimes, n)
	leg_stoch_discount_factors = include_yield_curve_basis(futurepaytimes, shortrates.get_yield_curve(), discount_curve, timegrid, n, sr_stoch_discount_factors)

	#distinguish between amortizing and fixed notional
	if isfloat(notional):
		notional = float(notional) #make sure it is floating type
		fixedpayment = freq*rate*notional #freq is used as approx for yearfrac, since daycount is not taken into account
		fixedvalues = fixedpayment * leg_stoch_discount_factors #This is a matrix, possibly containing only one column if only one payment is left. 

		#Add notional exchange, unless specified otherwise
		if nonotionalexchange is None:
			fixedvalues[:,-1] += notional*leg_stoch_discount_factors[:,-1]

		fixedvalues = np.sum(fixedvalues, axis=1) #Each row is summed

	#if amortizing notional will not be float
	else: 
		amortizing = pd.read_excel(r'Input/Amortizing/' + notional + '.xlsx')
		amortizing_notional = np.array(amortizing['Notional'])
		amount_paytimes = len(futurepaytimes)
		amortizing_notional = amortizing_notional[-amount_paytimes:] #Only take notionals of payments yet to come, now length of amortizing_notional is equal to amount of columns of leg_stoch_discount_factors

		fixedpayment = freq*rate*amortizing_notional
		fixedvalues = fixedpayment * leg_stoch_discount_factors
		if nonotionalexchange is None:
			fixedvalues[:,-1] += amortizing_notional[-1]*leg_stoch_discount_factors[:,-1]

		fixedvalues = np.sum(fixedvalues, axis=1) #Each row is summed

	return(fixedvalues)


def reset_rate_calc(reset_rates, reset_times, maturity, freq, timegrid, n, shortrates, forward_curve):
	
	future_reset_times = reset_times[reset_times > timegrid[n]]
	
	if(len(future_reset_times)==0):
		return(reset_rates) #no more calculation has to be done if we are past last point of reset times in the simulation
	
	else:
		future_discount_times = np.append(future_reset_times, maturity)
		sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(future_discount_times, n) #Stochastic discount factors of the short rate curve 
		future_stoch_discount_factors = include_yield_curve_basis(future_discount_times, shortrates.get_yield_curve(), forward_curve, timegrid, n, sr_stoch_discount_factors) #Include det. basis of short rate curve - forward curve

		df_one = np.delete(future_stoch_discount_factors, -1, axis=1) #removes last column of matrix
		df_two = np.delete(future_stoch_discount_factors, 0, axis=1) #removes first column of matrixµ

		stoch_forward_rates = (df_one / df_two - 1) / np.diff(future_discount_times) #stochastic forward rates on payments yet to come

		reset_rates[:, (reset_rates.shape[1] - len(future_reset_times)):reset_rates.shape[1]] = stoch_forward_rates #replace final columns of reset rates with the new stoch forward rates

		return(reset_rates)


def floatvalue(notional, freq, spread, discount_curve, timegrid, n, shortrates, futurepaytimes, reset_rates, nonotionalexchange= None):
	#Calculate the stochastic discount factors on the future paytimes
	sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(futurepaytimes, n)
	leg_stoch_discount_factors = include_yield_curve_basis(futurepaytimes, shortrates.get_yield_curve(), discount_curve, timegrid, n, sr_stoch_discount_factors)

	#Determine the future reset rates
	future_reset_rates = reset_rates[:, (reset_rates.shape[1] - len(futurepaytimes)):reset_rates.shape[1]]

	#distinguish between amortizing and fixed notional
	if isfloat(notional):
		notional = float(notional) #make sure it is floating type
		floatingpayment = (future_reset_rates + spread) * freq * notional
		floatingvalues = floatingpayment * leg_stoch_discount_factors #piecewise multiplication of two matrices with same dimensions 

		#Add notional exchange, unless specified otherwise
		if nonotionalexchange is None:
			floatingvalues[:,-1] += notional*leg_stoch_discount_factors[:,-1]

		floatingvalues = np.sum(floatingvalues, axis=1) #Each row is summedµ

	#if amortizing notional will not be float
	else: 
		amortizing = pd.read_excel(r'Input/Amortizing/' + notional + '.xlsx')
		amortizing_notional = np.array(amortizing['Notional'])
		amount_paytimes = len(futurepaytimes)
		amortizing_notional = amortizing_notional[-amount_paytimes:] #Only take notionals of payments yet to come, now length of amortizing_notional is equal to amount of columns of leg_stoch_discount_factors

		floatingpayment = freq * (future_reset_rates + spread) * amortizing_notional
		floatingvalues = floatingpayment * leg_stoch_discount_factors
		if nonotionalexchange is None:
			floatingvalues[:,-1] += amortizing_notional[-1]*leg_stoch_discount_factors[:,-1]

		floatingvalues = np.sum(floatingvalues, axis=1) #Each row is summed

	return(floatingvalues)