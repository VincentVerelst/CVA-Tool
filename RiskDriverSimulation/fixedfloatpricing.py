from .riskdrivers import *
from .generalfunctions import *

def fixedpricing(legs, net_future_mtm, leg_input, timegrid, shortrates, fxrates):
	for leg in legs:
		
		paytimes = create_payment_times(leg_input['Freq'][leg], leg_input['StartDate'][leg], leg_input['EndDate'][leg], leg_input['ValDate'][leg])

	return(paytimes)
