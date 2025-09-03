import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = sns.load_dataset("iris")
print(df.head())
print(df.tail(6))

print(df.info())

print("df.shape: ",df.shape)

## Statistics about dataset
print(df.describe())

## checking for null values
print("\n checking for null values: \n")
print(df.isnull().sum())

## Univariate analysis: focus on variable 'species'
print(df.groupby('species').agg([np.mean, np.median]))

## Box plot 
plt.figure(figsize=(8,4)) 
sns.boxplot(x='species',y='sepal_width',data=df ,palette='YlGnBu')
plt.show()

## Distribution of particular species
sns.histplot(df['petal_width'], bins=40, color='b')
plt.title('petal width distribution plot')
plt.show()

## count of number of observation of each species
sns.countplot(x='species',data=df)
plt.show()

## Correlation map using a heatmap matrix
sns.heatmap(df.corr(numeric_only = True), linecolor='white', linewidths=1)
plt.show()

## Multivariate analysis – analyis between two or more variable or features
## Scatter plot to see the relation between two or more features like sepal length, petal length,etc
axis = plt.axes()
axis.scatter(df.sepal_length, df.sepal_width)
axis.set(xlabel='Sepal_Length (cm)', ylabel='Sepal_Width (cm)', title='Sepal-Length vs Width')
plt.show()

sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.show()

## From the above graph we can see that
# Iris-virginica has a longer sepal length while Iris-setosa has larger sepal width
# For setosa sepal width is more than sepal length
## Below is the Frequency histogram plot of all features
axis = df.plot.hist(bins=30, alpha=0.5)
axis.set_xlabel('Size in cm')
plt.show()

# From the above graph we can see that sepalwidth is longer than any other feature followed by petalwidth
## examining correlation
sns.pairplot(df, hue='species')
plt.show()

figure, ax = plt.subplots(2, 2, figsize=(8,8))
ax[0,0].set_title("sepal_length")
ax[0,0].hist(df['sepal_length'], bins=8)
ax[0,1].set_title("sepal_width")
ax[0,1].hist(df['sepal_width'], bins=6);
ax[1,0].set_title("petal_length")
ax[1,0].hist(df['petal_length'], bins=5);
ax[1,1].set_title("petal_width")
ax[1,1].hist(df['petal_width'], bins=5);
plt.show()

#--------------------------------------------------
'''
plot = sns.FacetGrid(df, hue="species")
plot.map(sns.histplot, "sepal_length").add_legend()
 
plot = sns.FacetGrid(df, hue="species")
plot.map(sns.histplot, "sepal_width").add_legend()
 
plot = sns.FacetGrid(df, hue="species")
plot.map(sns.histplot, "petal_length").add_legend()
''' 
plot = sns.FacetGrid(df, hue="species")
plot.map(sns.histplot, "petal_width").add_legend()

plt.show()
