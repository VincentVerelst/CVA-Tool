import numpy as np 
import pandas as pd 
from scipy import interpolate
from scipy.stats import norm
import scipy.optimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt


calibration_input = mcinput = pd.read_excel(r'Input/calibrationdetails.xlsx')

currency = calibration_input['Currency'][0]
swaption_vols_name = calibration_input['SwaptionVols'][0]
mean_reversion = calibration_input['MeanReversion'][0]
forward_curve_name = calibration_input['ForwardCurve'][0]
discount_curve_name = calibration_input['DiscountCurve'][0]

swaption_vols = pd.read_excel(r'Input/' + swaption_vols_name + '.xlsx') 
forward_curve = pd.read_excel(r'Input/Curves/' + forward_curve_name + '.xlsx')
discount_curve = pd.read_excel(r'Input/Curves/' + discount_curve_name + '.xlsx')


forward_curve_fun = interpolate.interp1d(forward_curve['TimeToZero'], forward_curve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
discount_curve_fun = interpolate.interp1d(discount_curve['TimeToZero'], discount_curve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate



#Calculates the atm rate for a swap of given annuity and floating leg specifications
def atm_swap_rate(annuity, floating_paytimes, floating_freq, forward_curve_fun, discount_curve_fun):
	floating_forward_df = np.power((1 + forward_curve_fun(floating_paytimes)), -floating_paytimes)
	floating_forward_rates = (floating_forward_df[0:(len(floating_forward_df)-1)] / floating_forward_df[1:len(floating_forward_df)] - 1) / floating_freq
	if floating_paytimes[0] == floating_freq:
		first_floating_forward_rate = forward_curve_fun(floating_paytimes[0])
	else:	
		first_df = np.power(1 + forward_curve_fun(floating_paytimes[0] - floating_freq), -(floating_paytimes[0] - floating_freq))
		second_df = np.power(1 + forward_curve_fun(floating_paytimes[0]), -(floating_paytimes[0]))
		first_floating_forward_rate = (first_df / second_df - 1) / floating_freq

	floating_forward_rates = np.append(first_floating_forward_rate, floating_forward_rates)

	float_leg_value =np.sum(floating_forward_rates * floating_freq * np.power((1 + discount_curve_fun(floating_paytimes)), -floating_paytimes))
	atm_rate = float_leg_value / annuity
	return(atm_rate)


def swaption_pricing_bachelier(sigma, strike, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, pay_receive='payer'):
	#Determines the prices of a payer swaption using the Bachelier model (= normal model) --> able to include receiver as well
	#Used to convert the normal sigma into a swaption price

	fixed_paytimes = np.arange(expiry+fixed_freq, expiry+tenor+fixed_freq, fixed_freq)
	floating_paytimes = np.arange(expiry+floating_freq, expiry+tenor+floating_freq, floating_freq)
	fixed_discount_factors = np.power((1 + discount_curve_fun(fixed_paytimes)), -fixed_paytimes)
	annuity = np.sum(fixed_discount_factors) * fixed_freq
	if strike == 'atm':
		#if ATM then bachelier formula simplifies greatly, since d = (forward - strike)/(sigma*sqrt(T)) is zero
		#no distinction between payer/receiver swaption
		swaption_value = sigma * np.sqrt(expiry) * norm.pdf(0) * annuity 
		
		return(swaption_value)
	#if a specific strike is specified other than 'atm' the atm strike has to be determined first 
	else:
		atm_rate = atm_swap_rate(annuity, floating_paytimes, floating_freq, forward_curve_fun, discount_curve_fun)
		
		#pricing function will now be different for payer and receiver swaptions
		d = (atm_rate - strike) / (sigma * np.sqrt(expiry))
		if pay_receive == 'payer':
			swaption_value = ((atm_rate - strike) * norm.cdf(d) + sigma * np.sqrt(expiry) * norm.pdf(d)) * annuity

		else:
			swaption_value = ((strike - atm_rate) * norm.cdf(-d) + sigma * np.sqrt(expiry) * norm.pdf(d)) * annuity

		return(swaption_value)


def break_even_point_calculate(c_vector, h_vector, discount_factors, zeta, y):
#function to calculate the break even point y* in the LGM model
#y is given as an input of the function, but in reality we have to find y so that this function is zero
	v_vector = (h_vector - h_vector[0]) * np.sqrt(zeta)
	f_vector = (discount_factors / discount_factors[0]) * np.exp(-np.power(v_vector,2) / 2 - v_vector * y)

	return(np.sum(c_vector * f_vector))


#Function to price ATM (only ATM!!) swaptions using LGM model
def swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, zeta):

	#Determine DF of the discount and forward curve at the expiry date
	first_discount_df = np.power(1 + discount_curve_fun(expiry), -expiry)
	first_forward_df = np.power(1 + forward_curve_fun(expiry), -expiry)

	fixed_paytimes = np.arange(expiry+fixed_freq, expiry+tenor+fixed_freq, fixed_freq)
	floating_paytimes = np.arange(expiry+floating_freq, expiry+tenor+floating_freq, floating_freq)

	fixed_discount_factors = np.power((1 + discount_curve_fun(fixed_paytimes)), -fixed_paytimes)
	floating_discount_factors = np.power((1 + discount_curve_fun(floating_paytimes)), -floating_paytimes)

	annuity = fixed_freq * np.sum(fixed_discount_factors)

	atm_rate = atm_swap_rate(annuity, floating_paytimes, floating_freq, forward_curve_fun, discount_curve_fun)
	
	#From this point on the fixed and floating discount factors and times may included the one at the expiry date itself
	fixed_paytimes = np.append(expiry, fixed_paytimes)
	floating_paytimes = np.append(expiry, floating_paytimes)

	fixed_discount_factors = np.append(first_discount_df, fixed_discount_factors)
	floating_discount_factors = np.append(first_discount_df, floating_discount_factors)
	#calculate necessary LGM parameters, names are equal to standard LGM notation
	h_vector = (1 - np.exp(-mean_reversion * fixed_paytimes)) / mean_reversion #definition of H(t)
	
	#calculate the c parameters for the lgm model
	c_vector = np.zeros(len(fixed_discount_factors))
	beta = fixed_discount_factors / np.power((1 + forward_curve_fun(fixed_paytimes)), -fixed_paytimes) # beta = DF_OIS / DF_LIBOR
	c_vector[0] = -beta[1] / beta[0]
	for i in range(1, len(c_vector) - 1):
		c_vector[i] = 1 - beta[i+1]/beta[i] + atm_rate * fixed_freq
	c_vector[-1] = 1 + atm_rate * fixed_freq 
	
	#We are going to find the root of F(y) by finding the minimum of |F(y)|^2, since this will give no error if initial guess is exactly in between two roots
	function_to_minimize = lambda y: np.power(np.abs(break_even_point_calculate(c_vector, h_vector, fixed_discount_factors, zeta, y)),2)

	y_min = scipy.optimize.fmin(func=function_to_minimize, x0=0, maxiter=1000, xtol=1E-6, ftol=1E-6, disp=False)


	#test uniqueness of y_min
	negative_c_indices = np.where(c_vector<0)[0] #returns all indices where C is strictly negative in an array

	#If all C's are positive 
	if len(negative_c_indices) == 0:
		test = c_vector[0]

	#Last element in the array is the last index where C is negative (and this can be index 0)
	else:
		last_negative_index = negative_c_indices[-1] #This is the last index where C is negative
		if last_negative_index == 0:
			test = c_vector[0]
		else:
			test = c_vector[0] + break_even_point_calculate(c_vector[1:last_negative_index], h_vector[1:last_negative_index], fixed_discount_factors[1:last_negative_index], zeta, y_min)



	if test <= 0:
		print('Unique')
		#if test parameter is negative, then y* is a unique solution of F(y) = 0 and swaption can be priced with following formula
		swaption_price = -np.sum(c_vector * fixed_discount_factors * norm.cdf(- (y_min + (h_vector - h_vector[0])*np.sqrt(zeta))))
	
	else:
		#If y* is not a unique solution of F(y) = 0, the swaption value has to be calculated numerically 
		print('Not unique, integrating numerically')
		
		integrand = lambda y: np.mazetamum(-break_even_point_calculate(c_vector, h_vector, fixed_discount_factors, zeta, y),0) * np.exp(-np.power(y,2)/2)

		swaption_price = fixed_discount_factors[0] / (np.sqrt(2 * np.pi)) * integrate.quad(integrand, -np.inf, np.inf) #integrate the integrand from -infinity to  infinity


	return(swaption_price)


def forwards_from_discount(paytimes, discount_factors, freq):
	#will return a vector with length equal to len(paytimes) - 1 containing the forward rates derived from the specified discount factors
	if len(paytimes) != len(discount_factors):
		print("Length paytimes must be equal to length discount factors")
		exit()

	first_discount_factors = discount_factors[0:len(discount_factors) - 1]
	second_discount_factors = discount_factors[1:len(discount_factors)]

	forwards = (first_discount_factors / second_discount_factors - 1) / freq

	return(forwards)


#Function to price ATM (only ATM!!) swaptions using LGM model
def new_swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, zeta):
	#payment schedules
	fixed_paytimes = np.arange(expiry, expiry+tenor+fixed_freq, fixed_freq)
	floating_paytimes = np.arange(expiry, expiry+tenor+floating_freq, floating_freq)

	#discount factors of the discounting curve on the payment schedules
	fixed_discount_factors = np.power(1 + discount_curve_fun(fixed_paytimes), -fixed_paytimes)
	floating_discount_factors = np.power(1 + discount_curve_fun(floating_paytimes), -floating_paytimes)
	#discount factors of the forward curve on the payment schedules
	fixed_forward_df_factors = np.power(1 + forward_curve_fun(fixed_paytimes), -fixed_paytimes)
	floating_forward_df_factors = np.power(1 + forward_curve_fun(floating_paytimes), -floating_paytimes)

	#forward rates on the floating paytimes for the forward curve and discounting curve
	floating_forward_rates_df = forwards_from_discount(floating_paytimes, floating_discount_factors, floating_freq)
	floating_forward_rates_fwd = forwards_from_discount(floating_paytimes, floating_forward_df_factors, floating_freq)

	#Basis spreadlets defined as the difference between the forward rates from the forward curve and the forward rates from the discount curve
	basis_spreadlets = floating_forward_rates_fwd - floating_forward_rates_df

	#Calculate atm swap rate
	annuity = fixed_freq * np.sum(fixed_discount_factors[1:len(fixed_discount_factors)])
	atm_rate = atm_swap_rate(annuity, floating_paytimes[1:len(floating_paytimes)], floating_freq, forward_curve_fun, discount_curve_fun)

	#Construct the cashflows for the floating leg following p.114 of Lichters, Stamm & Gallagher
	floating_c_vector = np.zeros(len(floating_paytimes))
	floating_c_vector[0] =  1 + (floating_discount_factors[1] / floating_discount_factors[0]) * floating_freq * basis_spreadlets[0]
	for i in range(1, len(floating_c_vector) - 1):
		floating_c_vector[i] =  (floating_discount_factors[i+1] / floating_discount_factors[i]) * floating_freq * basis_spreadlets[i]

	floating_c_vector[-1] = -1

	#Construct the cashflows for the fixed leg following p.114 of Lichters, Stamm & Gallagher 
	fixed_c_vector = np.ones(len(fixed_paytimes)-1) * -1 * atm_rate * fixed_freq
	
	#np.sum(discounted_fixed_c_vector) + np.sum(discounted_floating_c_vector) should give zero. Where fixed leg is discounted with fixed_discount_factors[1:len(fixed_discount_factors)] and floating leg is discount with floating_discount_factors
	
	#Generate combined cashflows (floating flows are mapped to the paytimes of the fixed flows)

	c_vector = np.zeros(len(fixed_paytimes))

	c_vector[0] = floating_c_vector[0]

	freq_ratio = fixed_freq / floating_freq 
	for i in range(1, len(fixed_paytimes)):
		floating_sum = 0
		#Mapping of the floating cashflows to the fixed cashflows. ONLY WORKS IF fixed_freq >= floating_freq, or equivalently if freq_ratio >= 1
		for j  in range(int(freq_ratio * (i-1) + 1), int(freq_ratio*i)+1):
			floating_sum += floating_c_vector[j] * (floating_discount_factors[j] / fixed_discount_factors[i])

		c_vector[i] = fixed_c_vector[i-1] + floating_sum

	#now np.sum(c_vector * fixed_discount_factors) should also be zero, --> c_vector represents the equivalant cashflows on fixed payment dates of the entire swap
	
	#Now we want that sum_i c_i * P(t_i) = zero, we will model the disount factor (zero coupon bond) P(t_i) with LGM
	#Define the LGM parameters and then minimize numerically
	h_vector = (1 - np.exp(-mean_reversion * fixed_paytimes)) / mean_reversion #definition of H(t)

	
	#We are going to find the root of F(y) by finding the minimum of |F(y)|^2, since this will give no error if initial guess is exactly in between two roots
	function_to_minimize = lambda y: np.power(np.abs(break_even_point_calculate(c_vector, h_vector, fixed_discount_factors, zeta, y)),2)

	y_min = scipy.optimize.fmin(func=function_to_minimize, x0=0, maxiter=1000, xtol=1E-6, ftol=1E-6, disp=False)

	#Pricing of the swaption
	strike_vector = fixed_discount_factors[1:len(fixed_discount_factors)] * np.exp(-h_vector[1:len(h_vector)] * y_min - 0.5 * np.power(h_vector[1:len(h_vector)],2) * zeta)

	capital_sigma_vector =(h_vector[1:len(h_vector)] - h_vector[0]) * np.sqrt(zeta)

	d_plus = (1 / capital_sigma_vector) * (np.log(fixed_discount_factors[1:len(fixed_discount_factors)] / (strike_vector * fixed_discount_factors[0])) + 0.5 * np.power(capital_sigma_vector,2))
	d_minus = (1 / capital_sigma_vector) * (np.log(fixed_discount_factors[1:len(fixed_discount_factors)] / (strike_vector * fixed_discount_factors[0])) - 0.5 * np.power(capital_sigma_vector,2))

	first_option_terms = np.sign(c_vector[1:len(c_vector)]) * fixed_discount_factors[1:len(fixed_discount_factors)] * norm.cdf(np.sign(c_vector[1:len(c_vector)]) * d_plus)
	second_option_terms = np.sign(c_vector[1:len(c_vector)]) * fixed_discount_factors[0] * norm.cdf(np.sign(c_vector[1:len(c_vector)]) * d_minus)

	option_values = first_option_terms - second_option_terms

	swaption_price = np.sum(np.abs(c_vector[1:len(c_vector)]) * option_values)
	#Once y* (y_min in code) had been found, we can use analytical pricing formula for swaption under LGM
	#swaption_price = -np.sum(c_vector * fixed_discount_factors * norm.cdf(- (y_min + (h_vector - h_vector[0])*np.sqrt(zeta))))

	#swaption_price = fixed_discount_factors[0] * norm.cdf(-y_min / np.sqrt(zeta)) - fixed_discount_factors[-1] * norm.cdf((-y_min -(h_vector[-1] - h_vector[0])*zeta)/(np.sqrt(zeta))) - np.sum(fixed_freq*atm_rate*fixed_discount_factors[1:len(fixed_discount_factors)-1]*norm.cdf((- y_min - (h_vector[1:len(h_vector)-1] - h_vector[0])*zeta)/(np.sqrt(zeta))))

	return(swaption_price)



def calibrate_swaption_lgm_zeta(swaption_vols, forward_curve_fun, discount_curve_fun, mean_reversion):
	#Find the zeta parameter of the LGM model, given the swaption details and HW mean reversion
	#Takes as input the swaption_vols dataframe and appends a column with the LGM zetas
	if 'Zeta' not in swaption_vols:
		swaption_vols.insert(len(swaption_vols.columns), 'Zeta', 0) #insert a column in the dataframe which will contain the zetas
	
	for i in range(0,len(swaption_vols['Expiry'])):

		#Step 1: find the value of the swaption by using the normal model
		swaption_price = swaption_pricing_bachelier(swaption_vols['sigma'][i], swaption_vols['Strike'][i], swaption_vols['Expiry'][i], swaption_vols['Tenor'][i], swaption_vols['FixedFreq'][i], swaption_vols['FloatFreq'][i], forward_curve_fun, discount_curve_fun, pay_receive='payer')

		# #Step 2: find the zeta that gives the LGM price closest to the swaption price 
		function_to_solve = lambda zeta: np.power((swaption_pricing_lgm(mean_reversion, swaption_vols['Expiry'][i], swaption_vols['Tenor'][i], swaption_vols['FixedFreq'][i], swaption_vols['FloatFreq'][i], forward_curve_fun, discount_curve_fun, zeta) - swaption_price),2)
		#zeta_min = scipy.optimize.fmin(func=function_to_solve, x0=1, maxiter=1000, xtol=1E-6, ftol=1E-6, disp=False)
		

		
		
		zeta_min = scipy.optimize.minimize(function_to_solve, x0= [1E-6], bounds=[(0, None)], method='L-BFGS-B', tol=1e-9)
		#zeta_zero = scipy.optimize.ridder(function_to_solve, a=0, b=200) 

		swaption_vols.loc[i,'Zeta'] = zeta_min.x

	return(swaption_vols)


def zeta_to_sigma(swaption_vols, mean_reversion):
	#Takes the dataframe containing the LGM zetas and adds an extra columns containing the hull-white vols, obtained by a bootstrapping method
	

	if 'HW sigma' not in swaption_vols:
		swaption_vols.insert(len(swaption_vols.columns), 'HW sigma', 0) #insert a column in the dataframe which will contain the hw volatilities

	#First HW sigma is obtained by using the inverse relation between sigma and zeta
	first_sigma_squared = swaption_vols['Zeta'][0] * 2 * mean_reversion / (np.exp(2 * mean_reversion * swaption_vols['Expiry'][0]) - 1)
	first_sigma = np.sqrt(first_sigma_squared)
	
	swaption_vols.loc[0,'HW sigma'] = first_sigma

	#Add other expiries they are obtained using a bootstrapping method
	for i in range(1,len(swaption_vols['Zeta'])):
		time_delta = swaption_vols['Expiry'][i] - swaption_vols['Expiry'][i-1] #Time difference between two consecutive expiries

		sigma_squared = 2 * mean_reversion * (swaption_vols['Zeta'][i] - swaption_vols['Zeta'][i-1]) / (np.exp(2 * mean_reversion * swaption_vols['Expiry'][i]) - np.exp(2 * mean_reversion * swaption_vols['Expiry'][i-1]))
		new_sigma = np.sqrt(sigma_squared)

		swaption_vols.loc[i,'HW sigma'] = new_sigma
	return(swaption_vols)


lgm_zetas = calibrate_swaption_lgm_zeta(swaption_vols, forward_curve_fun, discount_curve_fun, mean_reversion)

hw_vols = zeta_to_sigma(lgm_zetas, mean_reversion)
hw_vols.to_excel("Output/calibratedvolatilities.xlsx")



# expiry = 1/12
# tenor = 5
# fixed_freq = 0.5
# floating_freq = 0.25
# strike = 'atm'
# sigma = 37.27/10000
# zeta = 4.79728530185345E-06

# swaption_price = swaption_pricing_bachelier(sigma, strike, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, pay_receive='payer')

# print(swaption_price)


# print(swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, zeta))


# function_to_solve = lambda zeta: np.power(np.abs((swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, zeta) - swaption_price)),2)
# #zeta_min = scipy.optimize.fmin(func=function_to_solve, x0=1, maxiter=1000, xtol=1E-6, ftol=1E-6, disp=False)
			
		
# zeta_min = scipy.optimize.minimize(function_to_solve, x0= [1E-6], bounds=[(0, None)], method='L-BFGS-B', tol=1e-9)

# print(zeta_min.x)

# print(swaption_price)
# print(swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun, zeta_min.x))