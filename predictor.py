from sklearn.preprocessing import OneHotEncoder
import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

onehotdata=pickle.load(open('oneHotTotalfinal2.pkl'))
df=pd.read_csv('Final_Training_Data.csv')
df1=pd.read_csv('Final_Test_Data.csv')
tr=df.values
te=df1.values
tr1=tr[:,[1,2,5,7]]
te1=te[:,[1,2,5,7]]
Y=tr[:,-1]
tr2=onehotdata[:906]
te2=onehotdata[906:]
print "tr1.shape",tr1.shape
print "tr2.shape",tr2.shape
train=np.column_stack((tr1,tr2))
test=np.column_stack((te1,te2))

#PCA

# train = PCA(n_components=200).fit_transform(train)
# test = PCA(n_components=200).fit_transform(test)
# print "shape after PCA", train.shape, test.shape
'''
df=pd.read_csv('newData2.csv')
tr=df.values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
tr=imp.fit(tr).transform(tr)
train=tr[:,:-1]
Y=tr[:,-1]
clf = RandomForestClassifier(n_estimators=25)
'''
clf = RandomForestClassifier(n_estimators=25)
clf.fit(train,Y)
pred=clf.predict(test)
li=[]
for i in range(0,len(pred)):
	li.append("Stripe "+ str(int(pred[i])))

predli=np.array(li)

d2=pd.read_csv('Final_Test_DataOr.csv')
isin=d2.values
isin=isin[:,0]
ans=np.column_stack((isin,predli))

f=open('output29_5.csv','wb')
w=csv.writer(f,delimiter=',')
w.writerows(ans)
print "main prediction done"

x1,y1=train[700:],Y[700:]
xt,yt=train[:700],Y[:700]

clf.fit(x1,y1)
yy=clf.predict(xt)
print "validation prediction result", np.sum(yy==yt)*1.0/(len(yt)*1.0)



