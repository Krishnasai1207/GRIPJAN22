#!/usr/bin/env python
# coding: utf-8

# # Author : Krishna Sai
# 
# Data Science & Business Analytics Intern
# 
# Task 1 : Prediction using Supervised Machine Learning
# 
# GRIP - The Sparks Foundation
# 
# 
# In this task,the goal is to predict the percentage of a student based on the number of study hours using simple linear regression.

# In[41]:


#Libraries Used

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics


# # Collecting And Inspecting Data

# data = pd.read_csv("http://bit.ly/w-data")

# In[43]:


data.describe()


# In[44]:


data.head()


# In[45]:


data.info()


# # Data Pre-Processing

# In[46]:


x=data['Hours'].values.reshape(-1,1)
print(x)
print(x.shape)


# In[47]:


y=data['Scores'].values.reshape(-1,1)
print(y)
print(y.shape)


# # Visualizing Data

# In[48]:


# Plotting our data points on 2-D graph.

data.plot(x='Hours', y='Scores', style='o')
plt.title("Hours vs Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.show()


# # Defining and Training the model

# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)


# In[50]:


prediction=model.predict(x_test)


# In[51]:


plt.scatter(x_train, y_train,color='brown')
plt.plot(x_train, model.predict(x_train), color = 'violet')
plt.title('Hours vs Scores during Training set')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# #  Prediction Metrics

# In[52]:


print("MAE {}".format(metrics.mean_absolute_error(y_test,prediction)))
print("MSE {}".format(metrics.mean_squared_error(y_test,prediction)))


# # Visualizing the predicted results

# In[53]:


plt.scatter(x_test, y_test,color='brown')
plt.plot(x_test, model.predict(x_test), color = 'violet')
plt.title('Predicted : Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# # Actual vs predicted

# In[54]:


result=pd.DataFrame({'y':y_test.flatten(),'predicted':prediction.flatten()})
print(result)


# # Q. What will be the predicted score if a student studies for 9.25hr?

# In[55]:


print('if a student studies for 9.25hr , then he may score {}'.format(model.predict([[9.25]])))


# In[ ]:




