import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from scipy import interpolate





class RatesDriver:
	def __init__(self, name, yieldcurve, volatility, meanreversion):
		self.name = name
		self.volatility = volatility
		self.yieldcurve = yieldcurve
		self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
		self.yieldfun = interpolate.interp1d(self.yieldcurve['TimeToZero'], self.yieldcurve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
		self.meanreversion = meanreversion

	def get_name(self):
		return self.name

	def get_yieldcurve(self):
		return self.yieldcurve

	def get_volatility_frame(self):
		return self.volatility #to give to simulated object afterwards

	def get_volatility(self, time):
		return self.volatilityfun(time)

	def get_yield(self, time):
		return self.yieldfun(time)
	
	def get_meanreversion(self):
		return self.meanreversion

	def get_firstshortrate(self):
		return self.yieldcurve['ZeroRate'][1] #First short rate approximated with first nonzero zero rate

	def get_inst_fwd_rates(self, times):
		self.timedifferences = np.diff(times) #determine time intervals of timegrid, len(timedifferences) = len(times) - 1
		self.discount_factors = np.power(1 + self.get_yield(times), -times)
		self.instantaneous_forward_rates = (self.discount_factors[0:(len(self.discount_factors)-1)] / self.discount_factors[1:len(self.discount_factors)] - 1) / self.timedifferences #Approximate intstan. fwd rates with standard forward rates
		self.instantaneous_forward_rates = np.append(np.array(self.get_firstshortrate()), self.instantaneous_forward_rates) #Approximate first inst fwd rate (= first short rate) with first zero rate
		return self.instantaneous_forward_rates

	def get_beta(self, times):
		self.beta = self.get_inst_fwd_rates(times) + (np.power(self.get_volatility(times),2) / (np.power(self.get_meanreversion(),2))) * np.power(1 - np.exp(-self.get_meanreversion()*times),2)
		return self.beta 
		
class FXDriver:
	def __init__(self, name, spotfx, volatility):
		self.name = name
		self.volatility = volatility
		self.spotfx = spotfx
		self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility

	def get_name(self):
		return self.name

	def get_volatility_frame(self):
		return self.volatility

	def get_volatility(self, time):
		return self.volatilityfun(time)	

	def get_spotfx(self):
		return self.spotfx


class InflationDriver:
	def __init__(self, name, realrate, nominalrate, initialindex, realvolatility, nominalvolatiliy, indexvolatility, realmeanreversion, nominalmeanreversion):
		self.name = name
		self.realrate = realrate
		self.nominalrate = nominalrate
		self.initialindex = initialindex
		self.realvolatility = realvolatility
		self.nominalvolatility = nominalvolatility
		self.indexvolatility = indexvolatility
		self.realvolatilityfun = interpolate.interp1d(self.realvolatility['TimeToZero'], self.realvolatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
		self.nominalvolatilityfun = interpolate.interp1d(self.nominalvolatility['TimeToZero'], self.nominalvolatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
		self.indexvolatilityfun = interpolate.interp1d(self.indexvolatility['TimeToZero'], self.indexvolatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
		self.realratefun = interpolate.interp1d(self.realrate['TimeToZero'], self.realrate['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
		self.nominalratefun = interpolate.interp1d(self.nominalrate['TimeToZero'], self.nominalrate['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
		self.realmeanreversion = realmeanreversion
		self.nominalmeanreversion = nominalmeanreversion

	def get_name(self):
		return self.name

	def get_realvolatility(self, time):
		return self.realvolatilityfun(time)

	def get_nominalvolatility(self, time):
		return self.nominalvolatilityfun(time)

	def get_indexvolatility(self, time):
		return self.indexvolatilityfun(time)

	def get_realrate(self, time):
		return self.realratefun(time)

	def get_nominalrate(self, time):
		return self.nominalratefun(time)

	def get_initialindex(self):
		return self.initialindex

	def get_realmeanreversion(self):
		return self.realmeanreversion

	def get_nominalmeanreversion(self):
		return self.nominalmeanreversion


class EquityDriver:
	def _init__(self, name, initialindex, volatility):
		self.name = name
		self.initialindex = initialindex
		self.volatility = volatility
		self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility

	def get_name(self):
		return self.name

	def get_initialindex(self):
		return self.initialindex

	def get_volatility(self, time):
		return self.volatilityfun(time)


#After simumlation we need objects to determine the stochastic DF's etc.

class ShortRates:
	def __init__(self, name, simulated_rates, yieldcurve, volatility, meanreversion, timegrid):
		self.name = name
		self.simulated_rates = simulated_rates
		self.yieldcurve = yieldcurve
		self.volatility = volatility 
		self.meanreversion = meanreversion 
		self.timegrid = timegrid
		self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
		self.yieldfun = interpolate.interp1d(self.yieldcurve['TimeToZero'], self.yieldcurve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate

	def get_name(self):
		return self.name

	def get_simulated_rates(self):
		return self.simulated_rates

	def get_volatility(self, time):
		return self.volatilityfun(time)

	def get_yield(self, time):
		return self.yieldfun(time)
	
	def get_meanreversion(self):
		return self.meanreversion

		#times, timegrid, n, shortrates, simulation_amount
	def get_stochastic_affine_discount_factors(self, times, n):
		#ADD INST FWD RATES on timegrid as input 
		#times = future times on which you want to calculate stoch DFs (future times as seen from today!)
		#n = point in simulation, so right now we are on time = timegrid[n], so all times must be times > timegrid[n]
		libor_zc_times = self.get_yield(times)
		libor_df_times = np.power((1 + libor_zc_times), -times)
		libor_zc_n = self.get_yield(self.timegrid[n])
		libor_df_n = np.power((1 + libor_zc_n), -timegrid[n]) 
		#Calculate ln(a(t,T))
		first_term = np.ln(libor_df_times / libor_df_n)
		second_term = 
		pass

	def get_stochastic_zero_rate(self, times):
		###TO PROGRAM ####
		pass

class FXRates:
	def __init__(self, name, simulated_rates, spotfx, volatility):
		self.name = name
		self.simulated_rates = simulated_rates
		self.spotfx = spotfx
		self.volatility = volatility 
		self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility

	def get_name(self):
		return self.name

	def get_simulated_rates(self):
		return self.simulated_rates

	def get_spotfx(self):
		return self.spotfx

	def get_volatility(self, time):
		return self.volatilityfun(time)	