from .riskdrivers import *
from .generalfunctions import *

def fixedpricing(legs, net_future_mtm, leg_input, timegrid, shortrates, fxrates, simulation_amount, irinput):
	for leg in legs:
		
		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])

		maturity = paytimes[-1] #Yearfrac of the maturity

		#Create empty matrix to fill with future mtms
		future_mtm = np.zeros((simulation_amount, len(timegrid)))

		#Determine the right short rate object
		currency = leg_input['Currency'][leg]
		currency_name = irinput[currency][0]
		shortrates = next((x for x in shortrates if x.get_name() == currency_name), None)

		for n in range(0, len(timegrid[timegrid < maturity])):
			futurepaytimes = paytimes[paytimes > timegrid[n]]
			

		

	
