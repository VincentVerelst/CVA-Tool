import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from scipy import interpolate
from .riskdrivers import *

def better_yearfrac(x, y):
	#determines the yearfrac between two datetime dates, but also returns negative if y < x instead of error
	if y >= x:
		return(yf.yearfrac(x,y))
	else:
		return(-1*yf.yearfrac(y,x))
#Function to check if a matrix is positive definite (which is a requirement for Cholesky decomposition)
def is_positive_definite(x):
	return np.all(np.linalg.eigvals(x) > 0) #All eigenvalues must be larger than zero 

#If matrix is not positive definite, following function calculates the nearest positive definite matrix using Higham algorithm
def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearest_positive_definite(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

#Function that creates riskdrivers
def create_riskdrivers(irinput, fxinput, inflationinput, equityinput):
	#Define all the irdriver objects
	irdrivers = {}
	for i in range(0, irinput.count(axis=1)[0]):
		temp_name = irinput[irinput.columns[i]][0]
		temp_yieldcurve = pd.read_excel(r'Input/Curves/' + irinput[irinput.columns[i]][1] + '.xlsx')
		temp_volatility = pd.read_excel(r'Input/Volatility/' + irinput[irinput.columns[i]][2] + '.xlsx')
		temp_mean_reversion = irinput[irinput.columns[i]][3]
		irdriver = RatesDriver(temp_name, temp_yieldcurve, temp_volatility, temp_mean_reversion)
		irdrivers[irinput.columns[i]] = irdriver 

	fxdrivers = {}
	for i in range(0, fxinput.count(axis=1)[0]):
		temp_name = fxinput[fxinput.columns[i]][0]
		temp_spotfx = fxinput[fxinput.columns[i]][1]
		temp_volatility = pd.read_excel(r'Input/Volatility/' + fxinput[fxinput.columns[i]][2] + '.xlsx')
		fxdriver = FXDriver(temp_name, temp_spotfx, temp_volatility)
		fxdrivers[fxinput.columns[i]] = fxdriver

	inflationdrivers = {}
	for i in range(0, inflationinput.count(axis=1)[0]):
		temp_name = inflationinput[inflationinput.columns[i]][0]
		temp_currency = inflationinput[inflationinput.columns[i]][1]
		temp_inflation_rate = pd.read_excel(r'Input/Curves/' + inflationinput[inflationinput.columns[i]][2] + '.xlsx')
		temp_initial_index = inflationinput[inflationinput.columns[i]][3]
		temp_real_volatility = pd.read_excel(r'Input/Volatility/' + inflationinput[inflationinput.columns[i]][4] + '.xlsx')
		temp_index_volatility = pd.read_excel(r'Input/Volatility/' + inflationinput[inflationinput.columns[i]][5] + '.xlsx')
		temp_real_mean_reversion = inflationinput[inflationinput.columns[i]][6]
		temp_nominal_rate = pd.read_excel(r'Input/Curves/' + irinput[temp_currency][1] + '.xlsx') #Read in the nominal interest rate as specified in the IR input
		temp_nominal_volatility = pd.read_excel(r'Input/Volatility/' + irinput[temp_currency][2] + '.xlsx') #Read in the nominal interest rate volatility as specified in the IR input
		temp_nominal_mean_reversion = irinput[temp_currency][3]  #Read in the nominal interest rate volatility as specified in the IR input
		
		
		inflationdriver = InflationDriver(temp_name, temp_currency, temp_inflation_rate, temp_initial_index, temp_real_volatility, temp_real_mean_reversion, temp_nominal_rate, temp_nominal_volatility, temp_nominal_mean_reversion, temp_index_volatility)
		inflationdrivers[temp_currency] = inflationdriver #Give the inflation drivers the same name as the name of the IR driver they correspond to

	#Define all the equitydriver objects
	equitydrivers = {}
	for i in range(0, equityinput.count(axis=1)[0]):
		temp_name = equityinput[equityinput.columns[i]][0]
		temp_initial_index = equityinput[equityinput.columns[i]][1]
		temp_volatility = pd.read_excel(r'Input/Volatility/' + equityinput[equityinput.columns[i]][2] + '.xlsx')
		equitydriver = EquityDriver(temp_name, temp_initial_index, temp_volatility)
		equitydrivers[equityinput.columns[i]] = equitydriver

	return(irdrivers, fxdrivers, inflationdrivers, equitydrivers)


def ir_fx_simulate(timegrid, simulation_amount, irdrivers, fxdrivers, inflationdrivers, random_matrices, correlationmatrix):
	
	deltaT = np.diff(timegrid)
	irdrivers_list = list(irdrivers) #list of the keys ('domestic', 'foreign1', etc) of the irdrivers dictionary, used to find indices in correlation matrix
	fxdrivers_list = list(irdrivers)
	inflationdrivers_list = list(inflationdrivers)
	#deltaT = timegrid[2] - timegrid[1] #determine time intervals of timegrid
	n_irdrivers = len(irdrivers)
	n_fxdrivers = len(fxdrivers)
	n_inflationdrivers = len(inflationdrivers)

	#IR DRIVERS

	#Calculate deterministic beta factors so that r(t) = x(t) + beta(t), with x(t) ornstein uhlenbeck process
	betas = {}
	for i in irdrivers:
		beta = irdrivers[i].get_beta(timegrid)
		betas[i] = beta
	
	matrix_index = 0 #index of random and correlation matrix corresponding to the right risk driver (for loops will go over them in order, so each time it's increased by one)
	
	#Simulate the short rates
	short_rates = {}
	for j in irdrivers:
		ornstein_uhlenbeck = np.zeros((simulation_amount, len(timegrid)))
		sim_short_rates  = np.zeros((simulation_amount, len(timegrid)))
		sim_short_rates[:,0] = betas[j][0]

		if j == 'domestic':
			for i in range(0, len(timegrid) - 1):
				#Exact solution scheme for x(t)
				ornstein_uhlenbeck[:,i+1] = ornstein_uhlenbeck[:,i] * np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) + irdrivers[j].get_volatility(timegrid[i+1]) * random_matrices[matrix_index][:,i] * np.sqrt((1 - np.exp(-2 * irdrivers[j].get_meanreversion() * deltaT[i]))/(2 * irdrivers[j].get_meanreversion()))
				sim_short_rates[:, i+1] = ornstein_uhlenbeck[:,i+1] + betas[j][i+1]
				#Create shortrates object containing all necessary info needed later in pricing
			short_rates_object = ShortRates(irdrivers[j].get_name(), sim_short_rates, irdrivers[j].get_yieldcurve(), irdrivers[j].get_volatility_frame(), irdrivers[j].get_meanreversion(), timegrid, irdrivers[j].get_inst_fwd_rates(timegrid))
			short_rates[j] = short_rates_object
		
		else:
			for i in range(0, len(timegrid) - 1):
				#Exact solution scheme for x(t) but including the quanto adjustment to domestic risk neutral measure
				ornstein_uhlenbeck[:,i+1] = ornstein_uhlenbeck[:,i] * np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) + (irdrivers[j].get_volatility(timegrid[i+1]) * fxdrivers[j].get_volatility(timegrid[i+1]) * correlationmatrix[matrix_index][n_irdrivers + matrix_index - 1]/irdrivers[j].get_meanreversion()) * (np.exp(- irdrivers[j].get_meanreversion() * deltaT[i]) - 1) + irdrivers[j].get_volatility(timegrid[i+1]) * random_matrices[matrix_index][:,i] * np.sqrt((1 - np.exp(-2 * irdrivers[j].get_meanreversion() * deltaT[i]))/(2 * irdrivers[j].get_meanreversion()))
				sim_short_rates[:, i+1] = ornstein_uhlenbeck[:,i+1] + betas[j][i+1]
		
			#Create shortrates object containing all necessary info needed later in pricing
			short_rates_object = ShortRates(irdrivers[j].get_name(), sim_short_rates, irdrivers[j].get_yieldcurve(), irdrivers[j].get_volatility_frame(), irdrivers[j].get_meanreversion(), timegrid, irdrivers[j].get_inst_fwd_rates(timegrid))
			short_rates[j] = short_rates_object

		matrix_index += 1

	
	#FX DRIVERS

	fxspot_rates = {}
	for j in fxdrivers:
		spotfx = np.zeros((simulation_amount, len(timegrid)))
		spotfx[:,0] = fxdrivers[j].get_spotfx() 
		for i in range(0, len(timegrid) - 1):
			spotfx[:,i+1] = spotfx[:,i] + spotfx[:,i] * ( (short_rates['domestic'].get_simulated_rates()[:,i+1] - short_rates[j].get_simulated_rates()[:,i+1]) * deltaT[i] + fxdrivers[j].get_volatility(timegrid[i+1]) * np.sqrt(deltaT[i]) * random_matrices[matrix_index][:,i] )

		#Create fxrates object containing all necessary info needed later in pricing	
		spotfx_object = FXRates(fxdrivers[j].get_name(), spotfx, fxdrivers[j].get_spotfx(), fxdrivers[j].get_volatility_frame())
		fxspot_rates[j] = spotfx_object
		
		matrix_index += 1

	#INFLATION DRIVERS

	real_betas = {}
	for i in inflationdrivers:
		real_beta = inflationdrivers[i].get_beta(timegrid)
		real_betas[i] = real_beta

	inflation_rates = {}
	for j in inflationdrivers:
		real_ornstein_uhlenbeck = np.zeros((simulation_amount, len(timegrid)))
		sim_real_short_rates  = np.zeros((simulation_amount, len(timegrid)))
		sim_real_short_rates[:,0] = real_betas[j][0]

		inflationindex = np.zeros((simulation_amount, len(timegrid)))
		inflationindex[:,0] = inflationdrivers[j].get_initial_index()
		#if domestic, then only drift adjustment for real --> nominal measure and no adjustment in inflation process
		if j == 'domestic':
			for i in range(0, len(timegrid) - 1):
				#Simulate the real short rate
				#Exact solution scheme for x(t)
				real_ornstein_uhlenbeck[:,i+1] = real_ornstein_uhlenbeck[:,i] * np.exp(- inflationdrivers[j].get_real_mean_reversion() * deltaT[i]) + (inflationdrivers[j].get_real_volatility(timegrid[i+1]) * inflationdrivers[j].get_inflation_volatility(timegrid[i+1]) * correlationmatrix[matrix_index][1 + matrix_index]/inflationdrivers[j].get_real_mean_reversion()) * (np.exp(- inflationdrivers[j].get_real_mean_reversion() * deltaT[i]) - 1)  + inflationdrivers[j].get_real_volatility(timegrid[i+1]) * random_matrices[matrix_index][:,i] * np.sqrt((1 - np.exp(-2 * inflationdrivers[j].get_real_mean_reversion() * deltaT[i]))/(2 * inflationdrivers[j].get_real_mean_reversion()))
				sim_real_short_rates[:, i+1] = real_ornstein_uhlenbeck[:,i+1] + real_betas[j][i+1]

			matrix_index += 1
			#simulate the inflation index with FX process between nominal and real short rate
			for i in range(0, len(timegrid) - 1):
				inflationindex[:,i+1] = inflationindex[:,i] + inflationindex[:,i] * ( (short_rates[j].get_simulated_rates()[:,i+1] - sim_real_short_rates[:,i+1]) * deltaT[i] + inflationdrivers[j].get_inflation_volatility(timegrid[i+1]) * np.sqrt(deltaT[i]) * random_matrices[matrix_index][:,i] )
			
			matrix_index += 1

			inflation_rates_object = InflationRates(inflationdrivers[j].get_name(), short_rates[j].get_simulated_rates(), sim_real_short_rates, inflationindex, inflationdrivers[j].get_nominal_yieldcurve(), inflationdrivers[j].get_real_yieldcurve(), inflationdrivers[j].get_nominal_volatility_frame(), inflationdrivers[j].get_real_volatility_frame(), inflationdrivers[j].get_inflation_volatility_frame(), inflationdrivers[j].get_nominal_mean_reversion(), inflationdrivers[j].get_real_mean_reversion(), irdrivers[j].get_inst_fwd_rates(timegrid), inflationdrivers[j].get_inst_fwd_rates(timegrid), timegrid)
			inflation_rates[j] = inflation_rates_object
		#if foreign, then two drift adjustments in real process: real --> nominal and real --> domestic and one drift adjustment in inflation process: foreign --> domestic
		else:
			for i in range(0, len(timegrid) - 1):
				#Exact solution scheme for x(t)
				real_ornstein_uhlenbeck[:,i+1] = real_ornstein_uhlenbeck[:,i] * np.exp(- inflationdrivers[j].get_real_mean_reversion() * deltaT[i]) + ((inflationdrivers[j].get_real_volatility(timegrid[i+1]) * inflationdrivers[j].get_inflation_volatility(timegrid[i+1]) * correlationmatrix[matrix_index][1 + matrix_index] + inflationdrivers[j].get_real_volatility(timegrid[i+1]) * fxdrivers[j].get_volatility(timegrid[i+1]) * correlationmatrix[matrix_index][n_irdrivers + fxdrivers_list.index(j)] )/inflationdrivers[j].get_real_mean_reversion()) * (np.exp(- inflationdrivers[j].get_real_mean_reversion() * deltaT[i]) - 1)  + inflationdrivers[j].get_real_volatility(timegrid[i+1]) * random_matrices[matrix_index][:,i] * np.sqrt((1 - np.exp(-2 * inflationdrivers[j].get_real_mean_reversion() * deltaT[i]))/(2 * inflationdrivers[j].get_real_mean_reversion()))
				sim_real_short_rates[:, i+1] = real_ornstein_uhlenbeck[:,i+1] + real_betas[j][i+1]

			matrix_index += 1
			#simulate the inflation index with FX process between nominal and real short rate
			for i in range(0, len(timegrid) - 1):
				inflationindex[:,i+1] = inflationindex[:,i] + inflationindex[:,i] * ( (short_rates[j].get_simulated_rates()[:,i+1] - sim_real_short_rates[:,i+1] - inflationdrivers[j].get_inflation_volatility(timegrid[i+1]) * fxdrivers[j].get_volatility(timegrid[i+1])  * correlationmatrix[matrix_index][n_irdrivers + fxdrivers_list.index(j)]) * deltaT[i] + inflationdrivers[j].get_inflation_volatility(timegrid[i+1]) * np.sqrt(deltaT[i]) * random_matrices[matrix_index][:,i] )
			
			matrix_index += 1

			inflation_rates_object = InflationRates(inflationdrivers[j].get_name(), short_rates[j].get_simulated_rates(), sim_real_short_rates, inflationindex, inflationdrivers[j].get_nominal_yieldcurve(), inflationdrivers[j].get_real_yieldcurve(), inflationdrivers[j].get_nominal_volatility_frame(), inflationdrivers[j].get_real_volatility_frame(), inflationdrivers[j].get_inflation_volatility_frame(), inflationdrivers[j].get_nominal_mean_reversion(), inflationdrivers[j].get_real_mean_reversion(), irdrivers[j].get_inst_fwd_rates(timegrid), inflationdrivers[j].get_inst_fwd_rates(timegrid), timegrid)
			inflation_rates[j] = inflation_rates_object 

	return(short_rates, fxspot_rates, inflation_rates)


def mc_simulate_hwbs(irdrivers, fxdrivers, inflationdrivers, equitydrivers, correlationmatrix, timegrid, simulation_amount):
	#Count riskdrivers
	n_irdrivers = len(irdrivers)
	n_fxdrivers = len(fxdrivers)
	n_inflationdrivers = 2*len(inflationdrivers)  #each inflation drivers contains two risk drivers: real rate and inflation index
	n_equitydrivers = len(equitydrivers)
	n_totaldrivers = n_irdrivers + n_fxdrivers + n_inflationdrivers + n_equitydrivers

	#Generate antithetic correlated random paths using cholesky decomposition
	if not is_positive_definite(correlationmatrix):
		correlationmatrix = nearest_positive_definite(correlationmatrix)
		
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
	if startdate <= valdate:
		schedule = schedule[schedule > valdate] #remove all dates that are before the valuation date
	else:
		schedule = np.delete(schedule,0) #On the effective date only notional is exchanged, no payments happen, so take this into account when effective date is in the future
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

def fixedvalue(notional, freq, rate, discount_curve, timegrid, n, shortrates, futurepaytimes, notionalexchange):
	sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(futurepaytimes, n)
	leg_stoch_discount_factors = include_yield_curve_basis(futurepaytimes, shortrates.get_yield_curve(), discount_curve, timegrid, n, sr_stoch_discount_factors)

	#distinguish between amortizing and fixed notional
	if isfloat(notional):
		notional = float(notional) #make sure it is floating type
		fixedpayment = freq*rate*notional #freq is used as approx for yearfrac, since daycount is not taken into account
		fixedvalues = fixedpayment * leg_stoch_discount_factors #This is a matrix, possibly containing only one column if only one payment is left. 

		#Add notional exchange, unless specified otherwise
		if notionalexchange == 'yes':
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
		if notionalexchange == 'yes':
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


def floatvalue(notional, freq, spread, discount_curve, timegrid, n, shortrates, futurepaytimes, reset_rates, notionalexchange):
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
		if notionalexchange ==  'yes':
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
		if notionalexchange == 'yes':
			floatingvalues[:,-1] += amortizing_notional[-1]*leg_stoch_discount_factors[:,-1]

		floatingvalues = np.sum(floatingvalues, axis=1) #Each row is summed

	return(floatingvalues)



def stochastic_discount(net_future_mtm, shortrates_object, timegrid, final_discount_curve):
	#Integrate short rates using trapezium rule to obtain stochastic discount factors to today
	shortrates = shortrates_object.get_simulated_rates() 
	net_discounted_mtm = np.zeros((net_future_mtm.shape[0], len(timegrid)))
	
	net_discounted_mtm[:,0] = net_future_mtm[:,0]

	for n in range(1, net_discounted_mtm.shape[1]):
		new_shortrates = shortrates[:,0:(n+1)]
		new_timegrid = timegrid[0:n+1]
		sr_stoch_discount_factors = np.exp(-np.trapz(new_shortrates, new_timegrid))#Numerical integration of short rates with trapezium rule. Returns vector of length = simulation amount
		final_stoch_discount_factors = include_yield_curve_basis(timegrid[n], shortrates_object.get_yield_curve(), final_discount_curve, timegrid, 0, sr_stoch_discount_factors)
		net_discounted_mtm[:,n] = net_future_mtm[:,n] * final_stoch_discount_factors

	return(net_discounted_mtm)

def atm_swap_rate(times, tenor, timegrid, n, fixed_freq, float_freq, shortrates_fixed, discount_curve_fixed, shortrates_float, forward_curve_float, discount_curve_float):
#time is time at which we want to know the swap rate, so time >= timegrid[n] must hold
	times = np.array(times)
	atm_rates = np.zeros((shortrates_fixed.get_simulated_rates().shape[0], len(times))) #empty matrix to be filled with the atm swap rates

	for i in range(0, len(times)):

		time = times[i]

		future_fixed_paytimes = np.arange(time + fixed_freq, time + tenor + fixed_freq, fixed_freq)
		future_float_paytimes = np.arange(time + float_freq, time + tenor + float_freq, float_freq)

		#stochastic discount factors of the short rate curve on the future payment times for fixed and flaot
		fixed_sr_discount_factors = shortrates_fixed.get_stochastic_affine_discount_factors(future_fixed_paytimes, n)
		float_sr_discount_factors = shortrates_float.get_stochastic_affine_discount_factors(future_float_paytimes, n)

		#fixed discount factors
		fixed_discount_factors = include_yield_curve_basis(future_fixed_paytimes, shortrates_fixed.get_yield_curve(), discount_curve_fixed, timegrid, n, fixed_sr_discount_factors)

		#float forward rates 
		float_forward_df_factors = include_yield_curve_basis(future_float_paytimes, shortrates_float.get_yield_curve(), forward_curve_float, timegrid, n, float_sr_discount_factors)
		
		#Determine first (stochastic) discount factor on time, which is needed for the forward rates
		first_sr_forward_df_factor = shortrates_float.get_stochastic_affine_discount_factors(time, n)
		first_float_forward_df_factors = include_yield_curve_basis(time, shortrates_float.get_yield_curve(), forward_curve_float, timegrid, n, first_sr_forward_df_factor)
		float_forward_df_factors = np.append(first_float_forward_df_factors, float_forward_df_factors, axis=1)
		
		df_one = np.delete(float_forward_df_factors, -1, axis=1) #removes last column of matrix
		df_two = np.delete(float_forward_df_factors, 0, axis=1) #removes first column of matrix
		float_forward_rates = (df_one / df_two - 1) / float_freq #stochastic forward rates on payments yet to come

		#float discount factors
		float_discount_factors = include_yield_curve_basis(future_float_paytimes, shortrates_float.get_yield_curve(), discount_curve_float, timegrid, n, float_sr_discount_factors)

		#fixed leg value (annuity)
		fixed_values = fixed_discount_factors * fixed_freq #matrix of all future fixed payments 
		annuity = np.sum(fixed_values, axis=1) #vector of length simulation_amount

		#float leg value 
		float_values = float_discount_factors * float_forward_rates * float_freq #matrix times matrix times number
		float_value = np.sum(float_values, axis=1)

		atm_rate = float_value / annuity #this is then a vector of the stochastic atm swap rates for the given tenor at the point timegrid[n] starting at (forward) time time

		atm_rates[:,i] = atm_rate #nth column of the matrix to be filled  with the stochastic atm swap rates

	return(atm_rates)


def yoy_inflation_rates(paytimes, lag, timegrid, n, inflationrates):
	pass