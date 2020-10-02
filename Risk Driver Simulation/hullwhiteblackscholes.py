import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from scipy import interpolate


class HullWhiteBlackScholes:
	def __init__(self, irinput, fxinput, inflationinput, equityinput, correlation, simulation_amount, timegrid):
		self.irinput = irinput
		self.fxinput = fxinput
		self.inflationinput = inflationinput
		self.equityinput = equityinput
		self.correlation = correlation
		self.simulation_amount = simulation_amount
		self.timegrid = timegrid

		#Define all the irdriver objects
		self.irdrivers = []
		for i in range(0, self.irinput.count(axis=1)[0]):
			temp_name = self.irinput[irinput.columns[i]][0]
			temp_yieldcurve = pd.read_excel(r'Input/Curves/' + self.irinput[self.irinput.columns[i]][1] + '.xlsx')
			temp_volatility = pd.read_excel(r'Input/Volatility/' + self.irinput[self.irinput.columns[i]][2] + '.xlsx')
			temp_mean_reversion = self.irinput[irinput.columns[i]][3]
			self.irdriver = self.RatesDriver(temp_name, temp_yieldcurve, temp_volatility, temp_mean_reversion)
			self.irdrivers.append(self.irdriver)

		#Define all the fxdriver objects
		self.fxdrivers = []
		for i in range(0, self.fxinput.count(axis=1)[0]):
			temp_name = self.fxinput[fxinput.columns[i]][0]
			temp_spotfx = self.fxinput[fxinput.columns[i]][1]
			temp_volatility = pd.read_excel(r'Input/Volatility/' + self.fxinput[self.fxinput.columns[i]][2] + '.xlsx')
			self.fxdriver = self.FXDriver(temp_name, temp_spotfx, temp_volatility)
			self.fxdrivers.append(self.fxdriver)

		#Define all the inflationdriver objects
		self.inflationdrivers = []
		for i in range(0, self.inflationinput.count(axis=1)[0]):
			temp_name = self.inflationinput[inflationinput.columns[i]][0]
			temp_real_rate = pd.read_excel(r'Input/Curves/' + self.inflationinput[self.inflationinput.columns[i]][1] + '.xlsx')
			temp_nominal_rate = pd.read_excel(r'Input/Curves/' + self.inflationinput[self.inflationinput.columns[i]][2] + '.xlsx')
			temp_initial_index = self.inflationinput[inflationinput.columns[i]][3]
			temp_real_volatility = pd.read_excel(r'Input/Volatility/' + self.inflationinput[self.inflationinput.columns[i]][4] + '.xlsx')
			temp_nominal_volatility = pd.read_excel(r'Input/Volatility/' + self.inflationinput[self.inflationinput.columns[i]][5] + '.xlsx')
			temp_index_volatility = pd.read_excel(r'Input/Volatility/' + self.inflationinput[self.inflationinput.columns[i]][6] + '.xlsx')
			temp_real_mean_reversion = self.inflationinput[inflationinput.columns[i]][7]
			temp_nominal_mean_reversion = self.inflationinput[inflationinput.columns[i]][8]
			self.inflationdriver = self.InflationDriver(temp_name, temp_real_rate, temp_nominal_rate, temp_initial_index, temp_real_volatility, temp_nominal_volatility, temp_index_volatility, temp_real_mean_reversion, temp_nominal_mean_reversion)
			self.inflationdrivers.append(self.inflationdriver)

		#Define all the equitydriver objects
		self.equitydrivers = []
		for i in range(0, self.equityinput.count(axis=1)[0]):
			temp_name = self.equityinput[equityinput.columns[i]][0]
			temp_initial_index = self.equityinput[equityinput.columns[i]][1]
			temp_volatility = pd.read_excel(r'Input/Volatility/' + self.equityinput[self.equityinput.columns[i]][2] + '.xlsx')
			self.equitydriver = EquityDriver(temp_name, temp_initial_index, temp_volatility)
			self.equitydrivers.append(self.equitydriver)

	def get_irdrivers(self):
		return self.irdrivers

	def get_fxdrivers(self):
		return self.fxdrivers

	def get_inflationdrivers(self):
		return self.inflationdrivers

	def get_equitydrivers(self):
		return self.equitydrivers


	class RatesDriver:
		def __init__(self, name, yieldcurve, volatility, meanreversion):
			self.name = name
			self.volatility = volatility
			self.yieldcurve = yieldcurve
			self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility
			self.yieldfun = interpolate.interp1d(self.yieldcurve['TimeToZero'], self.yieldcurve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
			self.meanreversion = meanreversion


		def get_name(self):
			return self.name

		def get_volatility(self, time):
			return self.volatilityfun(time)

		def get_yield(self, time):
			return self.yieldfun(time)
		
		def get_meanreversion(self):
			return self.meanreversion

	class FXDriver:
		def __init__(self, name, spotfx, volatility):
			self.name = name
			self.volatility = volatility
			self.spotfx = spotfx
			self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility

		def get_name(self):
			return self.name

		def get_volatility(self, time):
			return self.volatilityfun(time)	

		def get_spotFX(self):
			return self.spotFX

	class InflationDriver:
		def __init__(self, name, realrate, nominalrate, initialindex, realvolatility, nominalvolatiliy, indexvolatility, realmeanreversion, nominalmeanreversion):
			self.name = name
			self.realrate = realrate
			self.nominalrate = nominalrate
			self.initialindex = initialindex
			self.realvolatility = realvolatility
			self.nominalvolatility = nominalvolatility
			self.indexvolatility = indexvolatility
			self.realvolatilityfun = interpolate.interp1d(self.realvolatility['TimeToZero'], self.realvolatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility
			self.nominalvolatilityfun = interpolate.interp1d(self.nominalvolatility['TimeToZero'], self.nominalvolatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility
			self.indexvolatilityfun = interpolate.interp1d(self.indexvolatility['TimeToZero'], self.indexvolatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility
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