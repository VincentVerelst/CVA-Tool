from .riskdrivers import *
from .generalfunctions import *
from progressbar import ProgressBar


#Calculate a matrix of convexity adjustment CMS rates (same principle as stochastic floating rate matrix)
def cms_reset_calc(reset_rates, reset_times, tenor, freq, timegrid, n, shortrates, discount_curve, forward_curve, convexity_adjustment):
	future_reset_times = reset_times[reset_times > timegrid[n]]
	convexityfun = interpolate.interp1d(convexity_adjustment['TimeToZero'], convexity_adjustment['Convexity'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate

	if(len(future_reset_times)==0):
		return(reset_rates) #no more calculation has to be done if we are past last point of reset times in the simulation
	
	else:
		convexity = np.array(convexityfun(future_reset_times))
		stoch_forward_rates = atm_swap_rate(future_reset_times, tenor, timegrid, n, 0.5, freq, shortrates, discount_curve, shortrates, forward_curve, discount_curve)
		#add deterministic convexity adjustment
		stoch_reset_rates = stoch_forward_rates + convexity #convexity is a vector, so automatically each row of stoch_forward_rates will be added with this vector (length of vectir is equal to amount of columns of stoch_forward_rates)
		reset_rates[:, (reset_rates.shape[1] - len(future_reset_times)):reset_rates.shape[1]] = stoch_reset_rates #replace final columns of reset rates with the new stoch forward rates

		return(reset_rates)


#Pricing of a CMS leg 
def cmslegpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates, simulation_amount):
	for leg in legs:
		print('Pricing CMS leg ' + str(leg) + 'of total CMS legs ' + str(len(legs)))
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

		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < maturity]))):
			futurepaytimes = paytimes[paytimes > timegrid[n]]

			#Calculate stochastic reset rates
			reset_rates = cms_reset_calc(reset_rates, reset_times, tenor, forward_curve_freq, timegrid, n, shortrates, discount_curve_swaprate, forward_curve_swaprate, convexity_adjustment)

			#once the reset rates (= convexity adjusted forward CMS rates) have been determined, the pricing of the leg is identical to that of a floating leg
			cmsvalues = floatvalue(notional, freq, spread, discount_curve_leg, timegrid, n, shortrates, futurepaytimes, reset_rates, notional_exchange)

			future_mtm[:,n] = cmsvalues
		
		#convert with stochastic spot FX to domestic currency if it's a foreign leg
		if currency != 'domestic':
			future_mtm = future_mtm * fxrates[currency].get_simulated_rates()

		net_future_mtm += future_mtm

	return(net_future_mtm)


#Pricing of a CMS spread cap/floor: payoff = max(X - strike;0) if cap, max(strike - X;0) if floor, with X = CMS spread
def cmsspreadcapfloorpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates, simulation_amount):
	for leg in legs:
		print('Pricing CMS cap/floor leg ' + str(leg) + 'of total CMS legs ' + str(len(legs)))

		#load in all parameters
		currency = leg_input['Currency'][leg]
		freq = leg_input['Freq'][leg]
		swap_tenor_one = leg_input['SwapTenorOne'][leg] #If the spread is CMS_X - CMS_Y this is X
		first_reset_rate_one = leg_input['FirstResetRateOne'][leg] 
		convexity_adjustment_one = pd.read_excel(r'Input/Convexity/' + leg_input['ConvexityOne'][leg] +  '.xlsx')
		swap_tenor_two = leg_input['SwapTenorTwo'][leg] #If the spread is CMS_X - CMS_Y this is Y
		first_reset_rate_two = leg_input['FirstResetRateTwo'][leg] 
		convexity_adjustment_two = pd.read_excel(r'Input/Convexity/' + leg_input['ConvexityTwo'][leg] +  '.xlsx')
		strike = leg_input['Strike'][leg]
		leverage = leg_input['Leverage'][leg]
		spread = leg_input['Spread'][leg]
		forward_curve_swaprate = pd.read_excel(r'Input/Curves/' + leg_input['Forward Curve Swaprate'][leg] +  '.xlsx')
		forward_curve_freq = leg_input['Forward Curve Freq'][leg]
		discount_curve_swaprate = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve Swaprate'][leg] +  '.xlsx')
		discount_curve_leg = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve Leg'][leg] +  '.xlsx')
		normal_volatility = leg_input['NormalVol'][leg]
		notional = leg_input['Notional'][leg]
		cap_floor_flag = leg_input['CapOrFloor'][leg]

		switcher = {"cap":"call", "floor":"put"}
		call_put_flag = switcher[cap_floor_flag]
	

		#create payment schedule
		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])
		maturity = paytimes[-1]
		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right short rate object
		shortrates = shortrates_dict[currency] #Take shortrates object from list for which the name equals to the right currency

		#Initiate stochastic reset rate matrices for the two CMS tenors
		reset_rates_one = np.zeros((simulation_amount, len(paytimes)))
		reset_rates_one[:,0] = first_reset_rate_one #First reset rate was determined in the past before valuation date, so needs to be given as input
		
		reset_rates_two = np.zeros((simulation_amount, len(paytimes)))
		reset_rates_two[:,0] = first_reset_rate_two #First reset rate was determined in the past before valuation date, so needs to be given as input

		reset_times = paytimes[:-1] #Time schedule at which reset rates are determined. Payment for time at i is determined at i-1, so last payment is removed since all payments are decided from second to last payment

		#this is where the real pricing happens
		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < maturity]))):
			futurepaytimes = paytimes[paytimes > timegrid[n]]

			#Calculate the stochastic discount factors on the future paytimes
			sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(futurepaytimes, n)
			leg_stoch_discount_factors = include_yield_curve_basis(futurepaytimes, shortrates.get_yield_curve(), discount_curve_leg, timegrid, n, sr_stoch_discount_factors)
			
			#Calculate stochastic reset rates
			reset_rates_one = cms_reset_calc(reset_rates_one, reset_times, swap_tenor_one, forward_curve_freq, timegrid, n, shortrates, discount_curve_swaprate, forward_curve_swaprate, convexity_adjustment_one)

			reset_rates_two = cms_reset_calc(reset_rates_two, reset_times, swap_tenor_two, forward_curve_freq, timegrid, n, shortrates, discount_curve_swaprate, forward_curve_swaprate, convexity_adjustment_two)

			#Only future reset rates are needed
			future_reset_rates_one = reset_rates_one[:, (reset_rates_one.shape[1] - len(futurepaytimes)):reset_rates_one.shape[1]]
			future_reset_rates_two = reset_rates_two[:, (reset_rates_two.shape[1] - len(futurepaytimes)):reset_rates_two.shape[1]]

			future_forward_cms_spreads = leverage * (future_reset_rates_one - future_reset_rates_two + spread)

			caplet_floorlet_values = bachelier_model_call_put_price(futurepaytimes, future_forward_cms_spreads, strike, normal_volatility, timegrid, n, leg_stoch_discount_factors, call_put_flag)
			
			cap_floor_values = np.sum(caplet_floorlet_values, axis=1) * notional

			future_mtm[:,n] = cap_floor_values

	#convert with stochastic spot FX to domestic currency if it's a foreign leg
		if currency != 'domestic':
			future_mtm = future_mtm * fxrates[currency].get_simulated_rates()

		net_future_mtm += future_mtm

	return(net_future_mtm)