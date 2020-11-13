from .riskdrivers import *
from .generalfunctions import *

#Pricing of a ZC inflation leg
def zcinflationpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates_dict, inflationrates_dict, simulation_amount):
	for leg in legs:
		
		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Calculate the yearfrac (relative to the valuation date) on which the inflation indices are fixed
		starttime = better_yearfrac(leg_input['ValDate'][leg], leg_input['StartDate'][leg])
		paytime = better_yearfrac(leg_input['ValDate'][leg], leg_input['EndDate'][leg])
		lag = leg_input['Lag'][leg]
		notional = leg_input['Notional'][leg]

		#select the right risk driver object from the dictionaries
		currency = leg_input['Currency'][leg]
		shortrates = shortrates_dict[currency]
		fxrates = fxrates_dict[currency]
		inflationrates = inflationrates_dict[currency]

		#the base inflation index is set to the user input, if however the startdate is in the future, this will be overridden in the for loop
		stochastic_inflation_index_starttime = np.ones(simulation_amount).reshape((simulation_amount,1))
		stochastic_inflation_index_starttime = stochastic_inflation_index_starttime * leg_input['Base Index'][leg]

		#for n in range(0, len(timegrid[timegrid < paytime])):
		n = 0
		#stochastic spot inflation index, i.e. the inflation index (like RPI) at time = timegrid[n]
		spot_inflation_index = inflationrates.get_simulated_inflation_index()[:,n].reshape((simulation_amount,1))

		#determine the (potentially) stochastic (forward) inflation index on the starttime
		if (starttime - lag) > timegrid[n]:
			#stochastic stochastic zero coupon rates 
			stochastic_inflation_zc_rate_starttime = inflationrates.get_stochastic_inflation_rates(starttime, n)
			stochastic_inflation_index_starttime = spot_inflation_index * np.power(1 + stochastic_inflation_zc_rate_starttime , starttime - timegrid[n])
		
		#determine the (stochastic) forward inflation index on the paytime
		if (paytime - lag) > timegrid[n]:
			#stochastic stochastic zero coupon rates 
			stochastic_inflation_zc_rate = inflationrates.get_stochastic_inflation_rates(paytime, n)
			#stochastic forward inflation index at the paytime
			stochastic_inflation_index_paytime = spot_inflation_index * np.power(1 + stochastic_inflation_zc_rate, paytime - timegrid[n])

		#determine the stochastic discount factors at time timegrid n
		stochastic_discount_factor_sr = shortrates.get_stochastic_affine_discount_factors(paytime, n)
		#include potential basis to the discount curve
		stochastic_discount_factor = include_yield_curve_basis(paytime, shortrates.get_yield_curve(), pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve'][leg] +  '.xlsx'), timegrid, n, stochastic_discount_factor_sr)

		stochastic_value = (stochastic_inflation_index_paytime / stochastic_inflation_index_starttime) * notional * stochastic_discount_factor
		future_mtm[:,n] = np.sum(stochastic_value, axis=1) #sum is over one element, since ZC inflation is only one payment by definition, however this step is necessary for dimensions to match

		#convert with stochastic spot FX to domestic currency if it's a foreign leg 
		if currency != 'domestic':
			future_mtm = future_mtm * fxrates.get_simulated_rates()

		net_future_mtm += future_mtm


		return(stochastic_inflation_index_paytime)


#Pricing of a Year-on-Year inflation leg
def yoyinflationpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates_dict, inflationrates_dict, simulation_amount):
	pass