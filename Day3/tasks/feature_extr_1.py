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
#print(data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
print("after shuffling: ", data.head())
print("data.shape: ", data.shape)

array = data.values
X = array[:,0:46]
Y = array[:,46]
print(Y[:2])
# feature extraction
test = SelectKBest(score_func=f_classif, k=10)
fit = test.fit(X, Y)

# summarize scores
set_printoptions(precision=3)
print(fit.scores_, type(fit.scores_))
features = fit.transform(X)

# summarize selected features
print(features[0:5,:])

print(data.columns.to_numpy())

plt.rcParams["figure.figsize"] = (28,16)
plt.bar(data.columns.to_numpy()[:46], fit.scores_)
#plt.show()
plt.savefig("feature.png")
