import numpy as np 
import pandas as pd 
from numpy import genfromtxt


data = genfromtxt('Initial_Training_Data.csv', delimiter=',')


#df.apply(changeISIN, axis=1)
for i in range(1,len(data)):
	
	if int(data[i][8][-2:]) > 15:
             data[i][8]=1900+int(data[i][8][-2:])
	else:
	     data[i][8]=2000+int(data[i][8][-2:])
	
	data[i][12]=int(data[i][12][-2:])+2000
	

np.savetxt("newData2.csv", data, delimiter=",")		
	


