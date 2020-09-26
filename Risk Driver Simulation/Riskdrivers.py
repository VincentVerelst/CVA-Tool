import numpy as np
import pandas as pd
from scipy import interpolate

#Give errors if format is not right
volatilitydf = pd.read_excel(r'TestVolatility.xlsx')
yielddf = pd.read_excel(r'TestYieldCurve.xlsx')


class Ratesdriver:
	def __init__(self, name, volatility, yieldcurve):
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
	def __init__(self, name, volatility, spotFX):
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



domesticrate = Rates('EUR', volatilitydf, yielddf)
