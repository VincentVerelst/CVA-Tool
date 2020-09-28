import numpy as np 
import pandas as pd 
import riskdrivers

#Import everything

#Read in rates
IRinput = pd.read_excel(r'Runfiles/IRinput.xlsx')
IRinput = IRinput.dropna(1) #Drops all columns with NA values
IRinput = IRinput.drop(IRinput.columns[0], axis=1) #Drops first column
CurrencyAmount = IRinput.count(1)[0] #Count the amount of currencies

#Read in FX
FXinput = pd.read_excel(r'Runfiles/FXinput.xlsx')
FXinput = FXinput.dropna(1) #Drops all columns with NA values
FXinput = FXinput.drop(FXinput.columns[0], axis=1) #Drops first column

#Read in inflation (which is a combo of rates and FX)
Inflationinput = pd.read_excel(r'Runfiles/InflationInput.xlsx')
Inflationinput = Inflationinput.drop(Inflationinput.columns[0], axis=1) #Drops first column
Inflationinput = Inflationinput.dropna(1) #Drops all columns with NA values
InflationAmount = Inflationinput.count(1)[0] #Count the amount of currencies

Equityinput = pd.read_excel(r'Runfiles/EquityInput.xlsx')
Equityinput = Equityinput.drop(Equityinput.columns[0], axis=1) #Drops first column
Equityinput = Equityinput.dropna(1) #Drops all columns with NA values
EquityAmount = Equityinput.count(1)[0] #Count the amount of currencies

Correlationmatrix = pd.read_excel(r'Input/Correlation/correlationmatrix.xlsx', header=None,skiprows=1)
Correlationmatrix = Correlationmatrix.drop(Correlationmatrix.columns[0], axis=1)
Correlationmatrix = Correlationmatrix.values #Convert to numpy array (matrix)

num_rows, num_cols = Correlationmatrix.shape
n = CurrencyAmount + InflationAmount + EquityAmount
#Check if dimensions of correlation matrix are correct
if(num_rows != 2*n-1 or num_cols != 2*n-1):
	print("Error: enter correlation matrix with correct dimensions: (2n-1)x(2n-1), with n = amount of risk drivers")
	exit()

print(num_rows)


#Choleskycorr = np.linalg.cholesky(Correlationmatrix)
