#!/usr/bin/env python
# coding: utf-8

# # House Sales in King county,WA
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
# Goal is to explore the key influencers which cause House sales price to increase as well as to predict House sales price in 2016 based on the variables available in the dataset.

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
from scipy.stats import norm, skew
from sklearn.ensemble import RandomForestRegressor


# ## Reading the Data

# ##  Feature Description
# ID : Unique ID of each house sold
# Date : Date of House sale
# Price : Price of each sold house
# Bedrooms : Number of bedrooms
# Bathrooms : Number of bathrooms
# Sqft_living : Square footage of interior living space
# Sqft_lot : Square footage of land space
# Floors : Number of floors
# Waterfront : Variable that indicates whether the house overlooks waterfront or not
# View : An index from 0 to 4 of how good the view of the property is
# Condition : An index from 1 to 5 on the condition of the house
# Grade : An index from 1 to 13, where
#         1-3 falls short of building construction and design
#         7 has an avergae quality of construction and design
#         11-13 have higher quality of construction and design
# Sqft_above : Square footage of interior housing space that is above ground level
# Sqft_basement : Square footage of interior housing space that is below ground level
# Yr_built : Year house was built
# Yr_renovated : Year house was renovated
# Zipcode : Zipcode area the house is in
# Lat : Lattitude
# Long : Longitude
# Sqft_living15 :Square footage of interior living space for nearest 15 neighbours
# Sqft_lot15 :Square footage of land space for nearest 15 neighbours

# In[3]:


df = pd.read_csv(r'C:\Users\lsnee\OneDrive\Desktop\House data.csv')
print(df.shape)
print(df.info())


# ## Descriptive Statistics of Dataset

# In[4]:


print(df.describe())


# ## Are there any NULL values 

# In[5]:


print(df.isnull().any())


# ## Correlation between various features

# In[6]:


corr = df.corr()
plt.figure(figsize=(25,10))
Heatmap = sns.heatmap(corr,annot=True,linewidths=0.25,vmax=1.0, square=True, cmap="Greens", linecolor='k')


# * Price has high positive correlation with Sqft_living and moderate positive correlation with Bathrooms,Sqft_above,View,Grade and Sqft_living15
# * Price has low positive correlation with Bedrooms,floors,Sqft_basement and Yr_renovated
# * Price has non significant relationship with Sqft_lot,Year,Zipcode and Sqft_lot15

# ## Converting Date format

# In[8]:


import datetime
df['date']=pd.to_datetime(df['date'])
df = df.sort_values(by='date')
df


# ## Skewness of Data

# In[6]:


sns.distplot(df.skew(),color='blue',axlabel ='Skewness')


# ## Count of Bedrooms

# There is one house with 33 bedrooms which is very unique for any house

# In[22]:


print(df['bedrooms'].value_counts())
sns.countplot(df.bedrooms, order = df['bedrooms'].value_counts().index)


# ## Count of Unique Year built values

# In[32]:


print(df['yr_built'].value_counts().unique())
plt.figure(figsize=(16,9))
sns.countplot(df.yr_built, order = df['yr_built'].value_counts().index)
plt.tick_params(axis='x',which='major',labelsize=12)
plt.xticks(x='yr_built',rotation=90)
plt.tight_layout()


# ## Count of bathrooms

# There are 10 houses with 0 bathrooms

# In[29]:


print(df['bathrooms'].value_counts())
sns.countplot(df.bathrooms, order = df['bathrooms'].value_counts().index)
plt.tick_params(axis='x',which='major',labelsize=10)
plt.xticks(x='bathrooms',rotation=90)
plt.tight_layout()


# ## Count of Floors

# In[7]:


print(df['floors'].value_counts())
sns.countplot(df.floors, order = df['floors'].value_counts().index)
plt.tick_params(axis='x',which='major',labelsize=10)
plt.xticks(x='floors',rotation=90)
plt.tight_layout()


# ## Count of Grade

# In[35]:


print(df['grade'].value_counts())
sns.countplot(df.grade, order = df['grade'].value_counts().index)
plt.tick_params(axis='x',which='major',labelsize=10)
plt.xticks(x='grade',rotation=90)
plt.tight_layout()


# ## Scatter plot of Independent Variables

# In[9]:


sns.scatterplot(df.price,df.bedrooms)


# In[10]:


sns.scatterplot(df.price,df.bathrooms)


# In[11]:


sns.scatterplot(df.price,df.sqft_living)


# In[12]:


sns.scatterplot(df.price,df.sqft_lot)


# In[13]:


sns.scatterplot(df.price,df.floors)


# In[14]:


sns.scatterplot(df.price,df.yr_built)


# ## Removing Outliers from 'Bedrooms' column

# In[7]:


q1 = df['bedrooms'].quantile(0.25)
q3 = df['bedrooms'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['bedrooms']>=Lower_Fence)&(df['bedrooms']<=Upper_Fence)]
df.shape


# ## Removing Outlier from 'bathrooms' column

# In[8]:


q1 = df['bathrooms'].quantile(0.25)
q3 = df['bathrooms'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['bathrooms']>=Lower_Fence)&(df['bathrooms']<=Upper_Fence)]
df.shape


# ## Removing Outliers from Grade

# In[10]:


q1 = df['grade'].quantile(0.25)
q3 = df['grade'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['grade']>=Lower_Fence)&(df['grade']<=Upper_Fence)]
df.shape


# ## Removing Outliers from sqft_lot

# In[11]:


q1 = df['sqft_lot'].quantile(0.25)
q3 = df['sqft_lot'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['sqft_lot']>=Lower_Fence)&(df['sqft_lot']<=Upper_Fence)]
df.shape


# ## Removing Outliers from sqft_living

# In[12]:


q1 = df['sqft_living'].quantile(0.25)
q3 = df['sqft_living'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['sqft_living']>=Lower_Fence)&(df['sqft_living']<=Upper_Fence)]
df.shape


# ## Removing Outliers from sqft_above

# In[13]:


q1 = df['sqft_above'].quantile(0.25)
q3 = df['sqft_above'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['sqft_above']>=Lower_Fence)&(df['sqft_above']<=Upper_Fence)]
df.shape


# ## Removing Outliers sqft_living15

# In[14]:


q1 = df['sqft_living15'].quantile(0.25)
q3 = df['sqft_living15'].quantile(0.75)
IQR = q3-q1
IQR
Lower_Fence = q1 - (1.5 * IQR)
Upper_Fence = q3 + (1.5 * IQR)
df = df[(df['sqft_living15']>=Lower_Fence)&(df['sqft_living15']<=Upper_Fence)]
df.shape


# ## Defining Dependent and Independent Variables

# After analyzing the correlation between various variables and Price,variables that have significant linear relationship with Price were chosen as predictor variables.

# In[8]:


col = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','grade','view','sqft_above','yr_built','yr_renovated','sqft_living15','sqft_basement','waterfront']
X = df[col]
Y = df['price']
print(X.shape)
print(Y.shape)


# ## Divide the data into train and test sets

# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=100)


# ## Glance at the shape of the train and test sets

# In[10]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # Random Forest Regression : Fit the train sets into the model

# In[11]:


rf = RandomForestRegressor(n_estimators=100)
model_rf = rf.fit(X_train,Y_train)


# ## Predict Test variable

# In[12]:


y_rf_pred = model_rf.predict(X_test)
output= pd.DataFrame({"Actual":Y_test, "Predicted" : y_rf_pred})
output


# ## Metrics of the Model

# In[13]:


def mean_absolute_percentage_error(Y_test,y_rf_pred):
    Y_test, Y_pred = np.array(Y_test), np.array(y_rf_pred)
    return np.mean(np.abs((Y_test - y_rf_pred) / Y_test)) * 100
print('Mean Absolute Percentage Error(MAPE) : ', mean_absolute_percentage_error(Y_test,y_rf_pred))


# In[14]:


print('Mean Absolute Error : ', metrics.mean_absolute_error(Y_test,y_rf_pred))
print('Root Mean Squared Error : ', np.sqrt(metrics.mean_squared_error(Y_test,y_rf_pred)))
print('R Squared Value of the Model : ', metrics.r2_score(Y_test,y_rf_pred))


# ## Plotting the results

# In[15]:


plt.xlabel("y_test")
plt.ylabel("predicted values")
sns.regplot(Y_test,y_rf_pred)


# # Simple Linear Regression

# In[16]:


regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# ## Coefficients of the model

# In[17]:


print(regressor.intercept_)
print(regressor.coef_)


# ## Prediction

# In[18]:


Y_pred = regressor.predict(X_test)
Y_pred = pd.DataFrame(Y_pred, columns=['Predicted'])
Y_test = pd.DataFrame(Y_test,columns=['Actual'])
print(Y_pred,Y_test)


# * Predicting Selling Prices :
# Now that we have created the model, we can plug in the values of various variables and predict the House price.
# For example :
# Consider a House with
# Bedrooms = 2
# Bathrooms = 2
# Square footage of living space = 2500 sqft
# Square footage of the house lot = 5000 sqft
# Number of floors = 2
# Grade of the house = 7
# Noumber of views = 0
# Sqft_above = 2170 sqft
# Year house was built = 2005
# Year house was renovated = 0 (ie NO renovation)
# Sqft_living15 = 2500 sqft
# Sqft_basement = 0 (ie NO basement)
# Waterfront = 0
# Let's see what will be the House price based on above features

# In[42]:


col = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','grade','view','sqft_above','yr_built','yr_renovated','sqft_living15','sqft_basement','waterfront']
Feature_input = [[2,2,2500,5000,2,7,0,2170,2005,0,2500,0,0]]
for i,price in enumerate(regressor.predict(Feature_input)):
        print("Predicted selling price for Client's home: ${:,.2f}".format(price))


# Tableau Visualization : https://public.tableau.com/profile/neerajals#!/vizhome/Capstone2_15816359523670/Story1
