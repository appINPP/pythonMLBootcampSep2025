
import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

Signal_file = "feature_enginnering/vars_class1.csv"
Bkgd_file = "feature_enginnering/vars_class0.csv"

#select the variables you want to use for the algorithm training: 
columns = ["var10","var11","var20","var24","var29","var30","var31","var40","var41","var42","var43","var44","var45","var46",'label'] 

#get data from class 1
sig = pd.read_csv(Signal_file, usecols=columns)
print(sig.head())
print(sig.shape)
#get data from class 0
bkg = pd.read_csv(Bkgd_file, usecols=columns)
#print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
print(data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
#print("after shuffling: ", data.head())
print("data.shape: ", data.shape)

# save shuffled data
data.to_csv('shuffle_data.csv')
print(data.head())

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:85000,:14], data.iloc[:85000,14], test_size = 0.3, random_state = 100)
print("shape of training dataset: "," x: ",X_train.shape," y: ",y_train.shape)
print("X: ",X_train.head(), " y: ", y_train.head())

#Random Forest
mod_dt = RandomForestClassifier(n_estimators =100, oob_score=True, max_depth =3)
mod_dt.fit(X_train,y_train)

#prediction
predictions=mod_dt.predict(X_test) #predict: predicts the label
print("prediction: ",predictions)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(predictions,y_test)))

#see the importance of each predictor(variable) through its feature_importances_ attribute:
mod_dt.feature_importances_ 
print("feature importance: ", mod_dt.feature_importances_)
print(type(mod_dt.feature_importances_)) #print the type of mod_dt.feature_importances_ : it's a numpy array  

#Check the prediction results is through a confusion matrix:
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, predictions, labels=mod_dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=mod_dt.classes_)
disp.plot()
plt.show()

#Calculate and Plot ROC curve
from sklearn.metrics import roc_curve, auc
   
mod_dt.probability = True
probas = mod_dt.predict_proba(X_test) #predict probabilities: the predict_proba method returns the probabilities for each data point to belong in a class
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label =1)#assumes class 1 is the positive label
roc_auc  = auc(fpr, tpr)
 
fig2 = plt.figure(1,figsize=(10,15))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("RandomForest", roc_auc))
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc=0, fontsize='small')
fig2.savefig("ROCcurve.png")
plt.show()

# save the model to disk using pickle
import pickle

filename = 'finalized_model.sav'
pickle.dump(mod_dt, open(filename, 'wb'))
