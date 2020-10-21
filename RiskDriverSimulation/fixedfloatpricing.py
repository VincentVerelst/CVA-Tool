from .riskdrivers import *
from .generalfunctions import *

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

def fixedpricing(legs, net_future_mtm, leg_input, timegrid, shortrates, fxrates, simulation_amount, irinput):
	for leg in legs:
		
		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])

		maturity = paytimes[-1] #Yearfrac of the maturity

		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right short rate object
		currency = leg_input['Currency'][leg]
		currency_name = irinput[currency][0]
		shortrates = next((x for x in shortrates if x.get_name() == currency_name), None) #Take shortrates object from list for which the name equals to the right currency

		#Determine the parameters needed to price
		freq = leg_input['Freq'][leg]
		rate = leg_input['Rate'][leg]
		notional = leg_input['Notional'][leg]
		discount_curve = pd.read_excel(r'Input/Curves/' + leg_input['Discount Curve'][leg] +  '.xlsx')

		


		for n in range(0, len(timegrid[timegrid < maturity])):
			futurepaytimes = paytimes[paytimes > timegrid[n]]

			fixedvalues = fixedvalue(notional, freq, rate, discount_curve, timegrid, n, shortrates, futurepaytimes)

			future_mtm[:,n] = fixedvalues
			

		net_future_mtm += future_mtm

	return(net_future_mtm)

	
