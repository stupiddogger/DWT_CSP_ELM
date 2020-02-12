
# coding: utf-8

# load data


import scipy.io
import numpy as np
import mne
from scipy import signal
data=scipy.io.loadmat('dataset_BCIcomp1.mat')
data_test=data['x_test']
data_train=data['x_train']
label_train=data['y_train'].reshape(1,-1)-1
label=scipy.io.loadmat('y_test.mat')
label_test=label['y_test'].reshape(1,-1)-1
print(data_train.shape)
print(label_test.shape)
print(label_train.shape)
y_train=label_train[0]
y_test=label_test[0]
print(y_train.shape)
print(y_test.shape)
b,a=signal.butter(8,[(16/128),(64/128)],'bandpass')
buffer_x_test=signal.filtfilt(b,a,data_test,axis=0)
buffer_x_train=signal.filtfilt(b,a,data_train,axis=0)
print(buffer_x_test.shape)
all_x_train=np.transpose(buffer_x_train,[2,1,0])
all_x_test=np.transpose(buffer_x_test,[2,1,0])
x_train=all_x_train[:,0::2,448:896]
print(x_train.shape)
x_test=all_x_test[:,0::2,448:896]
print(x_test.shape)


# use DWT to reconstruct D2 and D3 bands


import pywt
db4=pywt.Wavelet('db4')
def Dwt(X):
    cA3,cD3,cD2,cD1 = pywt.wavedec(X,db4,mode='symmetric',level=3)
    return cA3,cD3,cD2,cD1
def cD3_features(x):
    Bands_D3=np.empty((x.shape[0],x.shape[1],448))
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            cA3,cD3,cD2,cD1=Dwt(x[i,ii,:])
            cA3=np.zeros(62)
            cD2=np.zeros(117)
            cD1=np.zeros(227)
            Bands_D3[i,ii,:]=pywt.waverec([cA3,cD3,cD2,cD1],db4)
    return Bands_D3
def cD2_features(x):
    Bands_D2=np.empty((x.shape[0],x.shape[1],448))
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            cA3,cD3,cD2,cD1=Dwt(x[i,ii,:])
            cA3=np.zeros(62)
            cD3=np.zeros(62)
            cD1=np.zeros(227)
            Bands_D2[i,ii,:]=pywt.waverec([cA3,cD3,cD2,cD1],db4)
    return Bands_D2
x_train_d3=cD3_features(x_train)
x_train_d2=cD2_features(x_train)
x_test_d3=cD3_features(x_test)
x_test_d2=cD2_features(x_test)
print(x_train_d3.shape)
print(x_test_d3.shape)
print(x_train_d2.shape)
print(x_test_d2.shape)


# use CSP to extract features


from mne.decoding import CSP
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
acc=[]
Csp1=CSP(n_components=2, reg=None, log=True, norm_trace=False)
Csp2=CSP(n_components=2, reg=None, log=True, norm_trace=False)
ss = preprocessing.StandardScaler()
X_train_d3=Csp1.fit_transform(x_train_d3,y_train)
X_test_d3=Csp1.transform(x_test_d3)
X_train_d2=Csp2.fit_transform(x_train_d2,y_train)
X_test_d2=Csp2.transform(x_test_d2)
print(X_train_d3.shape)
print(X_train_d2.shape)
print(X_test_d3.shape)
print(X_test_d2.shape)
X_train = ss.fit_transform(np.concatenate((X_train_d3,X_train_d2),axis=1))
X_test = ss.transform(np.concatenate((X_test_d3,X_test_d2),axis=1))
print(X_train.shape)
print(X_test.shape)


#use ELM as classifier


from hpelm import ELM
acc = []
elm=ELM(4,1)
elm.add_neurons(50,'sigm')
elm.train(X_train, y_train, "LOO")
y_pred=elm.predict(X_test)
print(len(y_pred))
for i in range(len(y_pred)):
    if y_pred[i]>=0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0        
print(y_test)
acc.append(accuracy_score(y_test, y_pred))
avg_acc=np.mean(acc)
print(avg_acc)


# use LDA as classifier


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_test)
print(y_pred)
acc=accuracy_score(y_test,y_pred)
print(acc)



#use svm as classifier
from sklearn import svm
clf=svm.SVC(C=0.5,kernel='rbf')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_test)
print(y_pred)
acc=accuracy_score(y_test,y_pred)
print(acc)

