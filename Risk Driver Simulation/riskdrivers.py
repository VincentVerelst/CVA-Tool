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