import numpy as np 
import pandas as pd 
from scipy import interpolate
from scipy.stats import norm



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

sigma = 42.69/10000
strike = 'atm'
expiry = 5
tenor = 5
fixed_freq = 1
floating_freq = 0.5

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
	annuity = np.sum(fixed_discount_factors)
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

		return(swaption_value, atm_rate)


def break_even_point_calculate(c_vector, h_vector, discount_factors, xi, y):
#function to calculate the break even point y^* in the LGM model
#y is given as an input of the function, but in reality we have to find y so that this function is zero
	v_vector = (h_vector - h_vector[0]) * np.sqrt(xi)
	f_vector = (discount_factors / discount_factors[0]) * np.exp(-np.power(v_vector,2) / 2 - v_vector * y)

	return(np.sum(c_vector * f_vector))


#Function to price ATM (only ATM!!) swaptions using LGM model
def swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun):

	fixed_paytimes = np.arange(expiry+fixed_freq, expiry+tenor+fixed_freq, fixed_freq)
	floating_paytimes = np.arange(expiry+floating_freq, expiry+tenor+floating_freq, floating_freq)

	fixed_discount_factors = np.power((1 + discount_curve_fun(fixed_paytimes)), -fixed_paytimes)
	floating_discount_factors = np.power((1 + discount_curve_fun(floating_paytimes)), -floating_paytimes)

	annuity = np.sum(fixed_discount_factors)

	atm_rate = atm_swap_rate(annuity, floating_paytimes, floating_freq, forward_curve_fun, discount_curve_fun)
	

	#calculate necessary LGM parameters, names are equal to standard LGM notation
	h_vector = (1 - np.exp(-mean_reversion * floating_paytimes)) / mean_reversion #definition of H(t)
	
	#calculate the c parameters for the lgm model
	c_vector = np.zeros(len(floating_discount_factors))
	beta = floating_discount_factors / np.power((1 + forward_curve_fun(floating_paytimes)), -floating_paytimes) # beta = DF_OIS / DF_LIBOR
	c_vector[0] = -beta[1] / beta[0]
	for i in range(1, len(c_vector) - 1):
		c_vector[i] = 1 - beta[i+1]/beta[i] + atm_rate * floating_freq
	c_vector[-1] = 1 + atm_rate * floating_freq 
	

	return(c_vector)



	

print(swaption_pricing_lgm(mean_reversion, expiry, tenor, fixed_freq, floating_freq, forward_curve_fun, discount_curve_fun))