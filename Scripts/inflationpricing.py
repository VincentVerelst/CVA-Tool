from .riskdrivers import *
from .generalfunctions import *
from progressbar import ProgressBar

#Pricing of a ZC inflation leg
def zcinflationpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates_dict, inflationrates_dict, simulation_amount):
	for leg in legs:
		print('Pricing ZC inflation leg ' + str(leg) + ' of total ZC inflation legs ' + str(len(legs)))
		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Calculate the yearfrac (relative to the valuation date) on which the inflation indices are fixed
		starttime = better_yearfrac(leg_input['ValDate'][leg], leg_input['StartDate'][leg])
		paytime = better_yearfrac(leg_input['ValDate'][leg], leg_input['EndDate'][leg])
		lag = leg_input['Lag'][leg]
		notional = leg_input['Notional'][leg]
		notionalexchange = leg_input['NotionalExchangeEnd'][leg]

		#select the right risk driver object from the dictionaries
		currency = leg_input['Currency'][leg]
		shortrates = shortrates_dict[currency]
		inflationrates = inflationrates_dict[currency]

		#the base inflation index is set to the user input, if however the startdate is in the future, this will be overridden in the for loop
		stochastic_inflation_index_starttime = np.ones(simulation_amount).reshape((simulation_amount,1))
		stochastic_inflation_index_starttime = stochastic_inflation_index_starttime * leg_input['Base Index'][leg]

		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < paytime]))):
		
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

			stochastic_value = (stochastic_inflation_index_paytime / stochastic_inflation_index_starttime - 1) * notional * stochastic_discount_factor
			if notionalexchange == 'yes':
				stochastic_value += notional * stochastic_discount_factor #Add notional to the final value if the notional is exchanged

			future_mtm[:,n] = np.sum(stochastic_value, axis=1) #sum is over one element, since ZC inflation is only one payment by definition, however this step is necessary for dimensions to match

		
		#convert with stochastic spot FX to domestic currency if it's a foreign leg 
		if currency != 'domestic':
			fxrates = fxrates_dict[currency]
			future_mtm = future_mtm * fxrates.get_simulated_rates()

		net_future_mtm += future_mtm


	return(net_future_mtm)

def yoyvalue(notional, spread, leverage, discount_curve, timegrid, n, shortrates, futurepaytimes, stoch_fwd_inflation_index_one, stoch_fwd_inflation_index_two, notionalexchange):
	#Calculate the stochastic discount factors on the future paytimes
	sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(futurepaytimes, n)
	leg_stoch_discount_factors = include_yield_curve_basis(futurepaytimes, shortrates.get_yield_curve(), discount_curve, timegrid, n, sr_stoch_discount_factors)
	
	#Determine the future stoch fwd inflation indices
	future_stoch_fwd_inflation_index_one = stoch_fwd_inflation_index_one[:, (stoch_fwd_inflation_index_one.shape[1] - len(futurepaytimes)):stoch_fwd_inflation_index_one.shape[1]]
	future_stoch_fwd_inflation_index_two = stoch_fwd_inflation_index_two[:, (stoch_fwd_inflation_index_two.shape[1] - len(futurepaytimes)):stoch_fwd_inflation_index_two.shape[1]]

	#future reset rates are then I_t / I_(t-1) - 1
	future_reset_rates = future_stoch_fwd_inflation_index_two / future_stoch_fwd_inflation_index_one - 1

	if isfloat(notional):
		notional = float(notional) #make sure it is floating type
		yoypayments = leverage * (future_reset_rates + spread) * notional
		yoyvalues = yoypayments * leg_stoch_discount_factors #piecewise multiplication of two matrices with same dimensions 

		#Add notional exchange, unless specified otherwise
		if notionalexchange ==  'yes':
			yoyvalues[:,-1] += notional*leg_stoch_discount_factors[:,-1]

		yoyvalues = np.sum(yoyvalues, axis=1) #Each row is summed

	#if amortizing notional will not be float
	else: 
		amortizing = pd.read_excel(r'Input/Amortizing/' + notional + '.xlsx')
		amortizing_notional = np.array(amortizing['Notional'])
		amount_paytimes = len(futurepaytimes)
		amortizing_notional = amortizing_notional[-amount_paytimes:] #Only take notionals of payments yet to come, now length of amortizing_notional is equal to amount of columns of leg_stoch_discount_factors

		yoypayments = leverage * (future_reset_rates + spread) * amortizing_notional
		yoyvalues = yoypayments * leg_stoch_discount_factors
		if notionalexchange == 'yes':
			yoyvalues[:,-1] += amortizing_notional[-1]*leg_stoch_discount_factors[:,-1]

		yoyvalues = np.sum(yoyvalues, axis=1) #Each row is summed

	return(yoyvalues)

#Pricing of a Year-on-Year inflation leg
def yoyinflationpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates_dict, inflationrates_dict, simulation_amount):
	for leg in legs:
		print('Pricing yoy inflation leg ' + str(leg) + ' of total yoy inflation legs ' + str(len(legs)))
		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right risk driver objects
		currency = leg_input['Currency'][leg]
		shortrates = shortrates_dict[currency]
		inflationrates = inflationrates_dict[currency]

		#Determine the parameters needed to price
		base_index_one = leg_input['Base Index One'][leg]
		base_index_two = leg_input['Base Index Two'][leg]
		lag = leg_input['Lag'][leg]
		freq = leg_input['Reset Frequency'][leg]
		spread = leg_input['Spread'][leg]
		leverage = leg_input['Leverage'][leg]
		notional_exchange = leg_input['NotionalExchangeEnd'][leg]
		notional = leg_input['Notional'][leg]
	    
		discount_curve = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve'][leg] +  '.xlsx')

		#the paytimes are always assumed equal to reset times (since discounting effect in CVA is negligible compared to reset frequency effect)
		paytimes = create_payment_times(freq, leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])
		#As yoy payments are determined at I_year / I_(year-1) we also need times
		paytimes_minus_one = paytimes - freq
		#maturity in years
		maturity = paytimes[-1] #Yearfrac of the maturity

		#stochastic forward inflation indices one and two. reset rate = I_2 / I_1 for every payment date
		stoch_fwd_inflation_index_one = np.zeros((simulation_amount, len(paytimes)))
		stoch_fwd_inflation_index_two = np.zeros((simulation_amount, len(paytimes)))
		#The first inflation index of one has already been determined in the past and has to be given as an input (lag already included, but this is by default in Bloomberg)
		stoch_fwd_inflation_index_one[:,0] = base_index_one
		stoch_fwd_inflation_index_one[:,1] = base_index_two #The second base index must also be included in the first matrix 
		#The first inflation index of two MAY have alraedy been determined in the past, if not, then it will be overridden in the stoch_fwd_inflation_index function sincen then paytimes[0] - lag > timegrid[0] = 0
		stoch_fwd_inflation_index_two[:,0] = base_index_two
		
		#loop over timegrid
		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < maturity]))):
			#Calculate the new stochastic forward inflation indices
			stoch_fwd_inflation_index_one = stoch_fwd_inflation_index(stoch_fwd_inflation_index_one, paytimes_minus_one, timegrid, n, inflationrates, lag)
			stoch_fwd_inflation_index_two = stoch_fwd_inflation_index(stoch_fwd_inflation_index_two, paytimes, timegrid, n, inflationrates, lag)
			#determine the future paytimes relative to the point in the simulation
			future_paytimes = paytimes[paytimes > timegrid[n]]
			#calculate the stochastic yoy leg values
			yoyvalues = yoyvalue(notional, spread, leverage, discount_curve, timegrid, n, shortrates, future_paytimes, stoch_fwd_inflation_index_one, stoch_fwd_inflation_index_two, notional_exchange)
			
			future_mtm[:,n] = yoyvalues

		#convert with stochastic spot FX to domestic currency if it's a foreign leg
		if currency != 'domestic':
			#fxrates = fxrates_dict[currency]
			future_mtm = future_mtm * fxrates_dict[currency].get_simulated_rates()

		net_future_mtm += future_mtm

	return(net_future_mtm)

def giltvalue(base_index, base_coupon, freq, notional_exchange, notional, discount_curve, stoch_fwd_inflation_indices, timegrid, n, future_paytimes, shortrates):
	#Calculate the stochastic discount factors on the future paytimes
	sr_stoch_discount_factors = shortrates.get_stochastic_affine_discount_factors(future_paytimes, n)
	leg_stoch_discount_factors = include_yield_curve_basis(future_paytimes, shortrates.get_yield_curve(), discount_curve, timegrid, n, sr_stoch_discount_factors)

    #Determine the future stoch fwd inflation indices
	future_stoch_fwd_inflation_indices = stoch_fwd_inflation_indices[:, (stoch_fwd_inflation_indices.shape[1] - len(future_paytimes)):stoch_fwd_inflation_indices.shape[1]]

	#determine payment values (I_fwd / I_base) * coupon * freq * notional
	giltpayments = future_stoch_fwd_inflation_indices / base_index * base_coupon * freq * notional 
	if notional_exchange == 'yes':
		giltpayments[:,-1] += notional 

    #stochastically discount the payments 
	giltpaymentspv = giltpayments * leg_stoch_discount_factors

	giltvalues = np.sum(giltpaymentspv, axis=1) #Each row is summed

	return(giltvalues)


def giltpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates_dict, inflationrates_dict, simulation_amount):
	for leg in legs:
		print('Pricing gilt leg ' + str(leg) + ' of total gilt legs ' + str(len(legs)))
		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right risk driver objects
		currency = leg_input['Currency'][leg]
		shortrates = shortrates_dict[currency]
		inflationrates = inflationrates_dict[currency]

		#Determine the parameters needed to price
		base_index = leg_input['Base Index'][leg]
		first_pay_index = leg_input['First Payment Index'][leg]
		base_coupon = leg_input['Base Coupon'][leg]
		lag = leg_input['Lag'][leg]
		freq = leg_input['Reset Frequency'][leg]
		notional_exchange = leg_input['NotionalExchangeEnd'][leg]
	   
		notional = leg_input['Notional'][leg]
	    
		discount_curve = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve'][leg] +  '.xlsx')

		#the paytimes are always assumed equal to reset times (since discounting effect in CVA is negligible compared to reset frequency effect)
		paytimes = create_payment_times(freq, leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])
		#maturity in years
		maturity = paytimes[-1] #Yearfrac of the maturity

		#the stochastic forward inflation indices on the paytimes, the first may have already been determined in the past and has to be given as an input, if it is not given, it will be overridden
		stoch_fwd_inflation_indices = np.zeros((simulation_amount, len(paytimes)))
		stoch_fwd_inflation_indices[:,0] = first_pay_index 

		#loop over timegrid
		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < maturity]))):
			#Calculate the new stochastic forward inflation indices
			stoch_fwd_inflation_indices = stoch_fwd_inflation_index(stoch_fwd_inflation_indices, paytimes, timegrid, n, inflationrates, lag)
			#determine the future paytimes relative to the point in the simulation
			future_paytimes = paytimes[paytimes > timegrid[n]]
			#calculate the stochastic yoy leg values
			giltvalues = giltvalue(base_index, base_coupon, freq, notional_exchange, notional, discount_curve, stoch_fwd_inflation_indices, timegrid, n, future_paytimes, shortrates)
			
			future_mtm[:,n] = giltvalues

		#convert with stochastic spot FX to domestic currency if it's a foreign leg
		if currency != 'domestic':
			#fxrates = fxrates_dict[currency]
			future_mtm = future_mtm * fxrates_dict[currency].get_simulated_rates()

		net_future_mtm += future_mtm

	return(net_future_mtm)