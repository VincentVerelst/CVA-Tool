import numpy as np 
import pandas as pd 
import riskdrivers

#Import everything



IRinput = pd.read_excel(r'Runfiles/IRinput.xlsx')
IRinput = IRinput.dropna(1) #Drops all columns with NA values
IRinput = IRinput.drop(IRinput.columns[0], axis=1) #Drops first column

FXinput = pd.read_excel(r'Runfiles/FXinput.xlsx')
FXinput = FXinput.dropna(1) #Drops all columns with NA values
FXinput = FXinput.drop(FXinput.columns[0], axis=1) #Drops first column

Correlationmatrix = pd.read_excel(r'Input/Correlation/correlationmatrix.xlsx', header=None,skiprows=1)
Correlationmatrix = Correlationmatrix.drop(Correlationmatrix.columns[0], axis=1)
Correlationmatrix = Correlationmatrix.values #Convert to numpy array (matrix)


print(FXinput)


#Choleskycorr = np.linalg.cholesky(Correlationmatrix)
