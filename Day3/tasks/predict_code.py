
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

new_data = pd.read_csv("shuffle_data.csv",index_col=0)
print(new_data.head())

#Now use the stored weights to predict the unknown events(events from row 85000 onwards)
X_eval = new_data.iloc[85000:,:14] #the input variables 
y_eval = new_data.iloc[85000:,14] #the label column from the evaluation dataset
print("shape of the evaluation dataset: ",X_eval.shape, y_eval.shape)

# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
result = loaded_model.score(X_eval, y_eval)
y_pred=loaded_model.predict(X_eval)
print("Accuracy of predictions: ",result)

cn = [1, 0] #labels
#calculate confusion matrix for the evaluation dataset (unknown data)
cm2 = confusion_matrix(y_eval, y_pred, labels=loaded_model.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                              display_labels=loaded_model.classes_)
disp2.plot()
plt.show()


