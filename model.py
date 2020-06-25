#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #2: IMPORT LIBRARIES AND DATASET

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from jupyterthemes import jtplot
#jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)


# In[2]:


# read the csv file 
admission_df = pd.read_csv('Admission_Predict.csv')


# In[3]:


admission_df.head()


# In[4]:


# Let's drop the serial no.
admission_df.drop('Serial No.', axis=1, inplace=True)
admission_df.head()


# # TASK #3: PERFORM EXPLORATORY DATA ANALYSIS

# In[5]:


# checking the null values
admission_df.isnull().sum()


# In[6]:


# Check the dataframe information
admission_df.info()


# In[7]:


# Statistical summary of the dataframe
admission_df.describe()


# In[8]:


# Grouping by University ranking 
df_university = admission_df.groupby(by = 'University Rating').mean()
df_university


# # TASK #4: PERFORM DATA VISUALIZATION

# In[9]:


admission_df.hist(bins = 30, figsize = (20, 20), color='r')


# In[10]:


sns.pairplot(admission_df)


# In[11]:


corr_matrix = admission_df.corr()
plt.figure(figsize=(12,12,))
sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[ ]:





# In[ ]:





# # TASK #5: CREATE TRAINING AND TESTING DATASET

# In[12]:


admission_df.columns


# In[13]:


X = admission_df.drop(columns=['Chance of Admit'])


# In[14]:


y = admission_df['Chance of Admit']


# In[15]:


X.shape


# In[16]:


y.shape


# In[17]:


X = np.array(X)
y = np.array(y)


# In[18]:


y = y.reshape(-1,1)


# In[19]:


y.shape


# In[20]:


# scaling the data before training the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)


# In[21]:


scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


# In[22]:


# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# # TASK #6: TRAIN AND EVALUATE A LINEAR REGRESSION MODEL

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# In[24]:


linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)


# In[25]:


accuracy_LinearRegression = linear_regression_model.score(X_test, y_test)
accuracy_LinearRegression




# Decision tree builds regression or classification models in the form of a tree structure. 
# Decision tree breaks down a dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. 
# The final result is a tree with decision nodes and leaf nodes.
# Great resource: https://www.saedsayad.com/decision_tree_reg.htm

from sklearn.tree import DecisionTreeRegressor
decisionTree_model = DecisionTreeRegressor()
decisionTree_model.fit(X_train, y_train)


# In[34]:


accuracy_decisionTree = decisionTree_model.score(X_test, y_test)
accuracy_decisionTree


# In[35]:


# Many decision Trees make up a random forest model which is an ensemble model. 
# Predictions made by each decision tree are averaged to get the prediction of random forest model.
# A random forest regressor fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 


# In[36]:


from sklearn.ensemble import RandomForestRegressor
randomForest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
randomForest_model.fit(X_train, y_train)


# In[37]:


accuracy_randomforest = randomForest_model.score(X_test, y_test)
accuracy_randomforest



y_pred = linear_regression_model.predict(X_test)
plt.plot(y_test, y_pred, '^', color='r')


# In[39]:


y_predict_orig = scaler_y.inverse_transform(y_pred)
y_test_orig = scaler_y.inverse_transform(y_test)


# In[40]:


plt.plot(y_test_orig, y_predict_orig, '^', color='r')


# In[41]:


k = X_test.shape[1]
n = len(X_test)
n


# In[42]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# # EXCELLENT JOB! YOU SHOULD BE PROUD OF YOUR NEWLY ACQUIRED SKILLS

import pickle
pickle.dump(linear_regression_model, open('linear_regression_model.pkl', 'wb'))

model = pickle.load(open('linear_regression_model.pkl', 'rb'))
print(model.predict([[337, 118, 4, 4.5, 4.5, 9.67, 1]]))