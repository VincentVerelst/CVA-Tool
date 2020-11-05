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
        self.beta = self.get_inst_fwd_rates(times) + (np.power(self.get_volatility(times),2) / (2 * np.power(self.meanreversion,2))) * np.power(1 - np.exp(-self.meanreversion*times),2)
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
    def __init__(self, name, currency, inflation_rate, initial_index, real_volatility, real_mean_reversion, nominal_rate, nominal_volatility, nominal_mean_reversion, index_volatility):
        self.name = name
        self.currency = currency
        self.inflation_rate = inflation_rate
        self.initial_index = initial_index
        self.real_volatility = real_volatility
        self.real_mean_reversion = real_mean_reversion
        self.nominal_rate = nominal_rate
        self.nominal_volatility = nominal_volatility
        self.nominal_mean_reversion = nominal_mean_reversion
        self.index_volatility = index_volatility
        
        self.inflation_rate_fun = interpolate.interp1d(self.inflation_rate['TimeToZero'], self.inflation_rate['InflationRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
        self.real_volatility_fun = interpolate.interp1d(self.real_volatility['TimeToZero'], self.real_volatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
        self.nominal_rate_fun = interpolate.interp1d(self.nominal_rate['TimeToZero'], self.nominal_rate['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate
        self.nominal_volatility_fun = interpolate.interp1d(self.nominal_volatility['TimeToZero'], self.nominal_volatility['sigma'], 'next', fill_value='extrapolate') #piecewise constant interpolation of the volatility
        self.index_volatility_fun = interpolate.interp1d(self.index_volatility['TimeToZero'], self.index_volatility['sigma'], 'next', fill_value='extrapolate')
        
        #Create real_rate dataframe by interpolating the inflation rate and subtracting it from the nominal rate
        self.real_rate_dict = {'TimeToZero': self.nominal_rate['TimeToZero'], 'ZeroRate': self.nominal_rate['ZeroRate'] - self.inflation_rate_fun(self.nominal_rate['TimeToZero']) }
        self.real_rate = pd.DataFrame(data = self.real_rate_dict)
        self.real_rate_fun = interpolate.interp1d(self.real_rate['TimeToZero'], self.real_rate['ZeroRate'], 'linear', fill_value='extrapolate') #linear interpolation of the zero rate

        
    def get_name(self):
        return self.name

    def get_currency(self):
        return self.currency

    def get_nominal_yieldcurve(self):
        return self.nominal_rate

    def get_real_yieldcurve(self):
        return self.real_rate

    def get_inflation_rate(self):
        return self.inflation_rate

    def get_initial_index(self):
        return self.initial_index

    def get_nominal_volatility_frame(self):
        return self.nominal_volatility #to give to simulated object afterwards

    def get_real_volatility_frame(self):
        return self.real_volatility

    def get_inflation_volatility_frame(self):
        return self.index_volatility

    def get_nominal_volatility(self, time):
        return self.nominal_volatility_fun(time)

    def get_real_volatility(self, time):
        return self.real_volatility_fun(time)

    def get_inflation_volatility(self, time):
        return self.index_volatility_fun(time)

    def get_nominal_rate(self, time):
        return self.nominal_rate_fun(time)

    def get_real_rate(self, time):
        return self.real_rate_fun(time)
        
    def get_nominal_mean_reversion(self):
        return self.nominal_mean_reversion

    def get_real_mean_reversion(self):
        return self.real_mean_reversion
    
    def get_first_real_short_rate(self):
        return self.real_rate['ZeroRate'][1] #First short rate approximated with first nonzero zero rate

    #For simulation only real instand forward rates are needed since nominal are already handled in the IR driver class
    def get_inst_fwd_rates(self, times):
        self.timedifferences = np.diff(times) #determine time intervals of timegrid, len(timedifferences) = len(times) - 1
        self.discount_factors = np.power(1 + self.get_real_rate(times), -times)
        self.instantaneous_forward_rates = (self.discount_factors[0:(len(self.discount_factors)-1)] / self.discount_factors[1:len(self.discount_factors)] - 1) / self.timedifferences #Approximate intstan. fwd rates with standard forward rates
        self.instantaneous_forward_rates = np.append(np.array(self.get_first_real_short_rate()), self.instantaneous_forward_rates) #Approximate first inst fwd rate (= first short rate) with first zero rate
        return self.instantaneous_forward_rates

    #Same story: only real beta needed, nominal is handled in IR driver class
    def get_beta(self, times):
        self.beta = self.get_inst_fwd_rates(times) + (np.power(self.get_real_volatility(times),2) / (2 * np.power(self.real_mean_reversion,2))) * np.power(1 - np.exp(-self.real_mean_reversion*times),2)
        return self.beta 


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
    def __init__(self, name, simulated_rates, yieldcurve, volatility, meanreversion, timegrid, inst_fwd_rates):
        self.name = name
        self.simulated_rates = simulated_rates
        self.yieldcurve = yieldcurve
        self.volatility = volatility 
        self.meanreversion = meanreversion 
        self.timegrid = timegrid
        self.inst_fwd_rates = inst_fwd_rates
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

    def get_yield_curve(self):
        return self.yieldcurve
    
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
        libor_df_n = np.power((1 + libor_zc_n), -self.timegrid[n]) 
        #Calculate ln(a(t,T))
        first_term = np.log(libor_df_times / libor_df_n)
        second_term = self.inst_fwd_rates[n] * (1 / self.meanreversion) * (1 - np.exp(-self.meanreversion * (times - self.timegrid[n])))
        third_term = -0.5 * np.power(self.get_volatility(times), 2) * np.power((1 / self.meanreversion) * (1 - np.exp(-self.meanreversion * (times - self.timegrid[n]))), 2) * (1 / (2 * self.meanreversion)) * (1 - np.exp(- 2 * self.meanreversion * self.timegrid[n]))
        
        lna = first_term + second_term + third_term
        lna_matrix = np.tile(lna, (np.shape(self.simulated_rates)[0],1)) #Repeat the ln(a) vector (which has len = len(times)) in a matrix with amount of rows equal to simulated_short rates rows = simulation_amount

        b_factor = (1 / self.meanreversion) * (1 - np.exp(- self.meanreversion * (times - self.timegrid[n]))) #b(t,T)
        b_factor_matrix = np.transpose(np.tile(b_factor, (np.shape(self.simulated_rates)[0],1)))#Same as for lna: create a matrix with repeated b_factor in rows, but transposed because we still have to multiply it with array

        stoch_df_matrix = np.exp(lna_matrix - np.transpose(b_factor_matrix * self.simulated_rates[:,n]))

        return(stoch_df_matrix)

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