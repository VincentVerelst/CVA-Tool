import numpy as np 
import pandas as pd 


#Import everything
CurrencyAmount = 4

IRcols = [i for i in range(1,CurrencyAmount+1)] #Skip first column and only input columns with data
FXcols = [i for i in range(1,CurrencyAmount)] #Skip first column and only input columns with data

IRinput = pd.read_excel(r'IRinput.xlsx', usecols=IRcols)
FXinput = pd.read_excel(r'FXinput.xlsx', usecols=FXcols)
Correlationmatrix = pd.read_excel(r'correlationmatrix.xlsx', header=None,skiprows=1)
Correlationmatrix = Correlationmatrix.drop(Correlationmatrix.columns[0], axis=1)
Correlationmatrix = Correlationmatrix.values

Choleskycorr = np.linalg.cholesky(Correlationmatrix)
print(Choleskycorr)