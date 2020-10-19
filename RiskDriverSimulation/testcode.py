import numpy as np 
import pandas as pd 
import datetime
import math
import QuantLib as ql #Requires "pip install QuantLib" in Anaconda prompt
from QuantLib import *
import yearfrac as yf #Requires "pip install yearfrac" in Anaconda prompt
from generalfunctions import *

fixedleginput = pd.read_excel(r'fixedlegs.xlsx', skiprows=2, index_col=0)

cal = ql.UnitedKingdom()
start_date = ql.Date(fixedleginput['StartDate'][1].day, fixedleginput['StartDate'][1].month, fixedleginput['StartDate'][1].year)
end_date = ql.Date(fixedleginput['EndDate'][1].day, fixedleginput['EndDate'][1].month, fixedleginput['EndDate'][1].year)
frequency = fixedleginput['Freq'][1]*12
#schedule = ql.Schedule(start_date, end_date, ql.Period(str(int(frequency)) + 'M'), cal, ql.Following, ql.Following, ql.DateGeneration.Forward, False)
dates = ql.MakeSchedule(start_date, end_date, ql.Period(str(int(frequency)) + 'M'), calendar=cal, backwards=True, convention=ql.ModifiedFollowing)
schedule = np.array([ql_to_datetime(d) for d in dates])
schedule = schedule[schedule > fixedleginput['ValDate'][1]]
yearfrac = yf.yearfrac(fixedleginput['ValDate'][1], schedule)

print(yearfrac)
