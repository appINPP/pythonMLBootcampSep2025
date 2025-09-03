# Feature Selection with Univariate Statistical Tests
import pandas as pd
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

#load data
Signal_file = "vars_class1.csv"
Bkgd_file = "vars_class0.csv"

#get signal data
sig = pd.read_csv(Signal_file) #, usecols=columns)
#print(sig.head())
print(sig.shape)
#get Bkgd data
bkg = pd.read_csv(Bkgd_file) #, usecols=columns)
#print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
print(data0.head())
print(data0.shape)

plt.figure(figsize=(15,7))
plt.hist(sig["var10"], bins=100, label='var1-class1', alpha=0.5, color='r',log=True)
plt.hist(bkg["var10"], bins=100, label='var1-class0', alpha=0.5, color='b',log=True)
plt.xlabel('Var10', fontsize=25)
plt.ylabel('Number of entries', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 


