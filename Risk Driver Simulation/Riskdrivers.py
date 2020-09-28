import numpy as np
import pandas as pd
from scipy import interpolate


class HullWhiteBlackScholes:
	def __init__(self, IRinput, FXinput, correlation, n, timegrid):
		self.IRinput = IRinput
		self.FXinput = FXinput
		self.correlation = correlation
		self.n = n
		self.timegrid = timegrid

	class Ratesdriver:
		def __init__(self, name, yieldcurve, volatility):
			self.name = name
			self.volatility = volatility
			self.yieldcurve = yieldcurve
			self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility
			self.yieldfun = interpolate.interp1d(self.yieldcurve['TimeToZero'], self.yieldcurve['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate

		def get_name(self):
			return self.name

		def get_volatility(self, time):
			return self.volatilityfun(time)

		def get_yield(self, time):
			return self.yieldfun(time)
		

	class FXdriver:
		def __init__(self, name, spotFX, volatility):
			self.name = name
			self.volatility = volatility
			self.spotFX = spotFX
			self.volatilityfun = interpolate.interp1d(self.volatility['TimeToZero'], self.volatility['sigma'], 'zero', fill_value='extrapolate') #piecewise constant interpolation of the volatility

		def get_name(self):
			return self.name

		def get_volatility(self, time):
			return self.volatilityfun(time)	

		def get_spotFX(self):
			return self.spotFX

	class Inflationdriver:
		def __init__(self, name, realrate, nominalrate, initialindex, realvolatility, nominalvolatiliy, indexvolatility):
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


	class Equitydriver:
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