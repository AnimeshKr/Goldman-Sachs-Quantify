import numpy as np 
import pandas as pd 


def changeISIN(x):
	return int(x['ISIN'][4])


path='Initial_Training_Data.csv'
df=pd.read_csv(path)
col=df.columns

#df.apply(changeISIN, axis=1)
for i in range(0,len(df)):
	'''
	df[col[0]][i]=df[col[0]][i][4:]
	df[col[1]][i]=df[col[1]][i][9:]
	df[col[2]][i]=df[col[0]][i][12:]
	df[col[3]][i]=df[col[0]][i][8:]
	if df[col[4]][i] == 'N':
		df[col[4]][i]=1
	else:
		df[col[4]][i]=2
	df[col[5]][i]=df[col[5]][i][9:]
	if df[col[6]][i] == 'Y':
		df[col[6]][i]=1
	else:
		df[col[6]][i]=0
	'''
	if int(df['Issue_Date'][i][-2:]) > 15:
             df['Issue_Date'][i]=1900+int(df['Issue_Date'][i][-2:])
	else:
	     df['Issue_Date'][i]=2000+int(df['Issue_Date'][i][-2:])
	
	df['Maturity_Date'][i]=int(df['Maturity_Date'][i][-2:])+2000
	

df.to_csv('newData2.csv',index=False)		
	


