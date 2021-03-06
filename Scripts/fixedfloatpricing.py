from .riskdrivers import *
from .generalfunctions import *
from progressbar import ProgressBar


def fixedpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates, simulation_amount):
	for leg in legs:
		print('Pricing fixed leg ' + str(leg) + ' of total fixed legs ' + str(len(legs)))
		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])

		maturity = paytimes[-1] #Yearfrac of the maturity

		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right short rate object
		currency = leg_input['Currency'][leg]
		shortrates = shortrates_dict[currency] #Take shortrates object from list for which the name equals to the right currency

		#Determine the parameters needed to price
		freq = leg_input['Freq'][leg]
		rate = leg_input['Rate'][leg]
		notional = leg_input['Notional'][leg]
		notional_exchange = leg_input['NotionalExchangeEnd'][leg]
		discount_curve = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve'][leg] +  '.xlsx')

		

		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < maturity]))):
			futurepaytimes = paytimes[paytimes > timegrid[n]]

			fixedvalues = fixedvalue(notional, freq, rate, discount_curve, timegrid, n, shortrates, futurepaytimes, notional_exchange)

			future_mtm[:,n] = fixedvalues
		
		
		#convert with stochastic spot FX to domestic currency if it's a foreign leg 
		if currency != 'domestic':
			future_mtm = future_mtm * fxrates[currency].get_simulated_rates()

		net_future_mtm += future_mtm

	return(net_future_mtm)




def floatpricing(legs, net_future_mtm, leg_input, timegrid, shortrates_dict, fxrates, simulation_amount):
	for leg in legs:
		print('Pricing floating leg ' + str(leg) + ' of total floating legs ' + str(len(legs)))

		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])

		maturity = paytimes[-1] #Yearfrac of the maturity

		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right short rate object
		currency = leg_input['Currency'][leg]
		shortrates = shortrates_dict[currency] #Take shortrates object from list for which the name equals to the right currency

		#Determine the parameters needed to price
		freq = leg_input['Freq'][leg]
		spread = leg_input['Spread'][leg]
		first_rate = leg_input['FirstResetRate'][leg]
		notional = leg_input['Notional'][leg]
		notional_exchange = leg_input['NotionalExchangeEnd'][leg]
		discount_curve = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve'][leg] +  '.xlsx')
		forward_curve = pd.read_excel(r'Input/Curves/' + leg_input['Forward Curve'][leg] +  '.xlsx')

		#Initiate stochastic reset rate matrix
		reset_rates = np.zeros((simulation_amount, len(paytimes)))
		reset_rates[:,0] = first_rate #First reset rate was determined in the past before valuation date, so needs to be given as input
		reset_times = paytimes[:-1] #Time schedule at which reset rates are determined. Payment for time at i is determined at i-1, so last payment is removed since all payments are decided from second to last payment

		pbar = ProgressBar()
		for n in pbar(range(0, len(timegrid[timegrid < maturity]))):
			futurepaytimes = paytimes[paytimes > timegrid[n]]

			#Calculate stochastic reset rates
			reset_rates = reset_rate_calc(reset_rates, reset_times, maturity, freq, timegrid, n, shortrates, forward_curve)

			floatvalues = floatvalue(notional, freq, spread, discount_curve, timegrid, n, shortrates, futurepaytimes, reset_rates, notional_exchange)

			future_mtm[:,n] = floatvalues
		
		#convert with stochastic spot FX to domestic currency if it's a foreign leg
		if currency != 'domestic':
			future_mtm = future_mtm * fxrates[currency].get_simulated_rates()

		net_future_mtm += future_mtm


	return(net_future_mtm)