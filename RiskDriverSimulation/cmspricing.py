from .riskdrivers import *
from .generalfunctions import *


def cms_reset_calc(reset_rates, reset_times, tenor, freq, timegrid, n, shortrates, discount_curve, forward_curve, convexity_adjustment):
	future_reset_times = reset_times[reset_times > timegrid[n]]
	convexityfun = interpolate.interp1d(convexity_adjustment['TimeToZero'], convexity_adjustment['Convexity'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate

	if(len(future_reset_times)==0):
		return(reset_rates) #no more calculation has to be done if we are past last point of reset times in the simulation
	
	else:
		convexity = np.array(convexityfun(future_reset_times))
		stoch_forward_rates = atm_swap_rate(future_reset_times, tenor, timegrid, n, 1, freq, shortrates, discount_curve, shortrates, forward_curve, discount_curve)
		#add deterministic convexity adjustment
		stoch_reset_rates = stoch_forward_rates + convexity #convexity is a vector, so automatically each row of stoch_forward_rates will be added with this vector (length of vectir is equal to amount of columns of stoch_forward_rates)
		reset_rates[:, (reset_rates.shape[1] - len(future_reset_times)):reset_rates.shape[1]] = stoch_reset_rates #replace final columns of reset rates with the new stoch forward rates

		return(reset_rates)

def cmslegpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates, simulation_amount):
	for leg in legs:

		currency = leg_input['Currency'][leg]
		freq = leg_input['Freq'][leg]
		spread = leg_input['Spread'][leg]
		first_rate = leg_input['Latest Index'][leg]
		notional = leg_input['Notional'][leg]
		notional_exchange = leg_input['NotionalExchangeEnd'][leg]
		tenor = leg_input['Swap Rate Tenor'][leg]
		convexity_adjustment = pd.read_excel(r'Input/Convexity/' + leg_input['Convexity'][leg] +  '.xlsx')
		discount_curve_swaprate = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve Swaprate'][leg] +  '.xlsx')
		discount_curve_leg = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve Leg'][leg] +  '.xlsx')
		forward_curve_swaprate = pd.read_excel(r'Input/Curves/' + leg_input['Forward Curve Swaprate'][leg] +  '.xlsx')
		forward_curve_freq = leg_input['Forward Curve Freq'][leg]


		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])
		maturity = paytimes[-1]
		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right short rate object
		shortrates = shortrates_dict[currency] #Take shortrates object from list for which the name equals to the right currency

		#Initiate stochastic reset rate matrix
		reset_rates = np.zeros((simulation_amount, len(paytimes)))
		reset_rates[:,0] = first_rate #First reset rate was determined in the past before valuation date, so needs to be given as input
		reset_times = paytimes[:-1] #Time schedule at which reset rates are determined. Payment for time at i is determined at i-1, so last payment is removed since all payments are decided from second to last payment

		for n in range(0, len(timegrid[timegrid < maturity])):
			futurepaytimes = paytimes[paytimes > timegrid[n]]

			#Calculate stochastic reset rates
			reset_rates = cms_reset_calc(reset_rates, reset_times, tenor, forward_curve_freq, timegrid, n, shortrates, discount_curve_swaprate, forward_curve_swaprate, convexity_adjustment)

			cmsvalues = floatvalue(notional, freq, spread, discount_curve_leg, timegrid, n, shortrates, futurepaytimes, reset_rates, notional_exchange)

			future_mtm[:,n] = cmsvalues
		
		#convert with stochastic spot FX to domestic currency if it's a foreign leg
		if currency != 'domestic':
			future_mtm = future_mtm * fxrates[currency].get_simulated_rates()

		net_future_mtm += future_mtm

		return(net_future_mtm)