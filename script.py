from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import pickle
from sklearn.cluster import KMeans


#test=pickle.load(open('completeTestdata.pkl'))
df=pd.read_csv('Final_Test_Data.csv')
test=df.values
df1=pd.read_csv('Final_Training_Data.csv')
train=df1.values
test1=test[:,[3,4,6,8,9,10,11,12,13,15]]
train1=train[:,[3,4,6,8,9,10,11,12,13,15]]
tr1=train[:,[1,2,5,7]]
te1=test[:,[1,2,5,7]]


total1=np.concatenate((train1,test1),axis=0)
tt2=np.concatenate((tr1,te1),axis=0)
tt3=np.column_stack((total1,tt2))

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
total1=imp.fit(total1).transform(total1)
tt3=imp.fit(tt3).transform(tt3)


#data=np.concatenate((train1,test1), axis=0)
est=KMeans(init='k-means++', n_clusters=10, n_init=10, n_jobs=-1)
est.fit(tt3)
label=est.predict(tt3)
data=np.column_stack((total1,label))

est=KMeans(init='k-means++', n_clusters=15, n_init=10, n_jobs=-1)
est.fit(tt3)
label1=est.predict(tt3)
data1=np.column_stack((data,label1))

print "shape before encoding", total1.shape 
enc = OneHotEncoder()
enc.fit(data1)
print "done one hot fitting !"
totalOnehot= enc.transform(data1).toarray()
print "shape after one hot", totalOnehot.shape
pickle.dump(totalOnehot, open('oneHotTotalfinal2.pkl','wb'))


