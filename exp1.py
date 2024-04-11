#!/usr/bin/env python
# coding: utf-8

# # Descriptive Statistics

# In[54]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# In[55]:


df = pd.read_csv('uber.csv')


# In[56]:


df.head()


# In[57]:


df.fare_amount.median()


# ### Mean fares & passengers

# In[58]:


meanFares = np.mean(df["fare_amount"])
meanPassengers = np.mean(df["passenger_count"])
print(f"The mean of the fares column is: {round(meanFares,3)}")
print(f"The mean of the passenger count column is: {round(meanPassengers,3)}")


# ### Median fares & passengers

# In[59]:


medianFares = np.median(df["fare_amount"])
medianPassenger = np.median(df["passenger_count"])
print(f"The median of the fares column is: {round(medianFares,3)}")
print(f"The median of the passenger count column is: {round(medianPassenger,3)}")


# ### Mode of fares & passengers

# In[60]:


modeFares = stats.mode(df["fare_amount"])
modePassenger = stats.mode(df["passenger_count"])
print(f"The mean of the fares column is: {modeFares[0]}")
print(f"The mean of the passenger count column is: {modePassenger[0]}")


# ### Min, Max and Sum of fares & passengers

# In[61]:


minFares = np.min(df['fare_amount'])
minPassenger = np.min(df['passenger_count'])
print(f"The min fare is: {minFares}")
print(f"The min number passenger is: {minPassenger}")

maxFares = np.max(df['fare_amount'])
maxPassenger = np.max(df['passenger_count'])
print(f"The max fare is: {maxFares}")
print(f"The max number of passenger is: {maxPassenger}")

sumFares = np.sum(df['fare_amount'])
sumPassenger = np.sum(df['passenger_count'])
print(f"The sum of fares is: {sumFares}")
print(f"The sum of passenger count is: {sumPassenger}")


# ### Range, First Quartile, Third Quartile and Interquartile Range

# In[62]:


print("Fares: ")
print("Range: ",np.ptp(df["fare_amount"]))
print(df.fare_amount.describe())
print("\n")
print("Passenger Count: ")
print("Range: ",np.ptp(df["passenger_count"]))
print(df.passenger_count.describe())


# ### Standard Deviation & Variances of Fares and Passengers

# In[63]:


stdFares = np.std(df["fare_amount"])
print("Standard Deviation of Fares: ", round(stdFares,3))
varFares = np.var(df["fare_amount"])
print("Variance of Fares: ", round(varFares,3))

stdPassengers = np.std(df["passenger_count"])
print("Standard Deviation of Passengers: ", round(stdPassengers,3))
varPassengers = np.var(df["passenger_count"])
print("Variance of Fares: ", round(varPassengers,3))


# ### Correlation between Number of Passengers and Fares

# In[64]:


corr = np.corrcoef(df["fare_amount"],df["passenger_count"])
print(corr)


# ### Standard Error of Mean

# In[65]:


semFares = stats.sem(df["fare_amount"])
print("Standard Error of Mean of Fares: ", round(semFares,3))
semPassengers = stats.sem(df["passenger_count"])
print("Standard Error of Mean of Number of Passengers: ", round(semPassengers,3))


# ### Coefficient of Variation 

# In[66]:


cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 
covFares = cv(df["fare_amount"])
print("Coefficient of Variation of Fares: ", round(covFares,3))
covPass = cv(df["passenger_count"])
print("Coefficient of Variation of Passenger count: ", round(covPass,3))


# ### Null values in every column

# In[67]:


df.isnull().sum()


# ### Total Count of each Column

# In[68]:


df.count()


# ### Cumulative Percentage

# In[69]:


cpFares = 100 * (df['fare_amount'].cumsum()/df['fare_amount'].sum()) 
print("Cumulative Percentage of Fares: ", cpFares)


# ### Trimmed Mean

# In[70]:


tmFares = stats.trim_mean(df["fare_amount"], 0.1)
print("Trimmed mean (0.1) of Fares is: ", tmFares)


# ### Sum of Squares

# In[71]:


ssFares = 0
ssPass = 0
for i in range (len(df['fare_amount'])):
    fare = df["fare_amount"][i]
    ssFares += (fare*fare)
    nPass = df["passenger_count"][i]
    ssPass += (nPass*nPass)
    
print("Sum of Square of fares is: ",ssFares)
print("Sum of Square of number of passengers is: ",ssPass)


# ### Skewness and Kurtosis

# In[72]:


# Skewness and Kurtosis
numeric_cols = df.select_dtypes(include=np.number).columns
skewness = df[numeric_cols].skew()
kurtosis = df[numeric_cols].kurtosis()

print("Skewness:")
print(skewness)
print("\nKurtosis:")
print(kurtosis)


# ### Plots

# ### Box and Whisker Plot

# In[73]:


df.boxplot(by="passenger_count", column="fare_amount",grid=False)


# ### Scatter Plot between fare amount and passenger count

# In[74]:


plt.scatter(df["fare_amount"], df["passenger_count"])


# ### Correlation Matrix

# In[75]:


# Convert date columns to datetime objects
df['date'] = pd.to_datetime(df['date'])
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Compute the correlation matrix for numeric columns only
numeric_cols = df.select_dtypes(include=np.number).columns
df_numeric = df[numeric_cols]
correlation_matrix = df_numeric.corr()

# Display the correlation matrix
print(correlation_matrix)


# In[76]:


f = plt.figure()
plt.matshow(correlation_matrix, fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

