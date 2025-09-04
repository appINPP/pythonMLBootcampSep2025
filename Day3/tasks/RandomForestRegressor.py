import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

#read the data:
file0 = "housing.csv"

'''
Columns of this dataset refer to:
longitude: A measure of how far west a house is; a higher value is farther west
latitude: A measure of how far north a house is; a higher value is farther north
housingMedianAge: Median age of a house within a block; a lower number is a newer building
totalRooms: Total number of rooms within a block
totalBedrooms: Total number of bedrooms within a block
population: Total number of people residing within a block
medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
medianHouseValue: Median house value for households within a block (measured in US Dollars)

Check the dataset: 
'''
sig = pd.read_csv(file0,index_col=0)
print(sig.head())
print(sig.shape)

#randomly shuffle the data
data = shuffle(sig, random_state=0) 
print("after shuffling: \n", data.head())
print("shuffled data.shape: ", data.shape)

print(data.info())

print(data.describe())

print("The column names of this dataframe are: ", data.columns)

'''
Exploratory Data Analysis (EDA)
'''
## Correlation map using a heatmap matrix
sns.heatmap(data.corr(numeric_only = True), linecolor='white', linewidths=1)
plt.show()

axis = plt.axes()
axis.scatter(data.MedInc, data.MEDV)
axis.set(xlabel='Median Area Income', ylabel='Median Price', title='scatter plot')
plt.show()

'''
Select input variables and label
'''
X = data[ ] """fill in appropriately the input variables """
y = data['MEDV']

'''
Split the data into a training set and a testing set
'''



'''
Random Forest Regressor
'''

