
# coding: utf-8

# In[1]:


from numpy import loadtxt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #not used?
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import matplotlib.pylab as plt
import time


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#DATA_DIR = "D:/2017_working/unearthed/output/"
#INPUT_DATA_DIR = "D:/2017_working/unearthed/"

INPUT_DATA_DIR = "/home/ubuntu/unearthed_water/"
DATA_DIR = "/home/ubuntu/unearthed_water/output/"
TARGET = "target"
NORM_TARGET = "norm_target"
UNAMED = "Unnamed: 0"
TIMESTAMP = "timestamp"


# In[5]:


df_test = pd.read_csv(INPUT_DATA_DIR+'publishable_test_set.csv')
print (type(df_test), df_test.shape)


# In[6]:


colnames = list(df_test)
colnames


# In[7]:


#convert df[TIMESTAMP] from string to datetime
df_test[TIMESTAMP] = pd.to_datetime(df_test[TIMESTAMP])


# In[8]:


df_test.dtypes


# In[9]:


print (type(df_test[TIMESTAMP].iloc[0]), df_test[TIMESTAMP].iloc[0])


# In[10]:


print (df_test[TIMESTAMP].iloc[0], df_test[TIMESTAMP].dtypes, type(df_test[TIMESTAMP].iloc[0]))

time_step = (df_test[TIMESTAMP].iloc[1]-df_test[TIMESTAMP].iloc[0])
print ("time_step(in seconds):", time_step, type(time_step))
time_start = df_test[TIMESTAMP].iloc[0]

print ("time_start = df[TIMESTAMP][0]", type(df_test[TIMESTAMP].iloc[0]), df_test[TIMESTAMP].iloc[0])


# In[11]:


#convert datetime to int 
#convert datetime to int since XGBoost does not handle datetime
#NB: df[TIMESTAMP][0] is set to zero with subsequent values = zero + seconds.
print ("converting df_test[TIMESTAMP] to int")
timestampNew = "timestamp_"
df_test[timestampNew] = df_test.apply(lambda x: int((x[TIMESTAMP] - time_start).total_seconds()), axis=1)
print ("converting df_test[TIMESTAMP] to int: completed.")
df_test.drop(labels=TIMESTAMP, axis=1, inplace=True)
df_test.rename(index=str, columns={timestampNew: TIMESTAMP}, inplace=True)

#NB: timestamp is now last column in dataframe


# In[12]:


df_test[TIMESTAMP].dtypes, type(df_test[TIMESTAMP]), df_test[TIMESTAMP].iloc[0], df_test[TIMESTAMP].iloc[1]


# In[13]:


null_count = df_test.isnull().sum()
print ("null_count:\n", null_count)
print ("total nulls:", null_count.sum())


# In[14]:


bad_list = ['No Data', 'I/O Timeout', 'Bad Input', 'Scan Off']


# In[15]:


print (df_test.shape)
cols_to_delete = []
for colname in colnames[1:]:
    #NB: TIMESTAMP column already converted above.
    try:
        badcount = []
        for bad in bad_list:
            badcount.append(df_test[colname].str.contains(bad).sum())
        print (colname, "\t", dict(zip(bad_list, badcount))  , df_test[colname].dtype)
        if badcount[0] == df_test.shape[0]:
            cols_to_delete.append(colname)
    except Exception as e: 
        print(e)
        print ("error", colname)
        #added this since reusing this code block for data exploration
print ("cols_to_delete:", cols_to_delete)


# In[16]:


print (df_test.shape)
df_test = df_test.drop(cols_to_delete, axis=1)
print (df_test.shape)


# In[17]:


colnames = list(df_test)
colnames


# In[18]:


#test converting columns from string values to floats with errors going to NaN
for colname in colnames[:-1]:
    print ("converting column:", colname)
    df_test[colname] = pd.to_numeric(df_test[colname], errors='coerce')


# In[19]:


#count NaN values in each column, then replace
null_count = df_test.isnull().sum()
print ("null_count:\n", null_count)
print ("total nulls:", null_count.sum())


# In[20]:


#convert null values to floats by filling data.
#df_test.fillna(method='bfill', axis=1, inplace=True)
for colname in colnames:
    print ("filling colname:", colname)
    df_test[colname].fillna(method='bfill', inplace=True)
#NB: weird error if attempt fillna on dataframe. refer github unfixed error.


# In[21]:


#count NaN values in each column, then replace
null_count = df_test.isnull().sum()
print ("null_count:\n", null_count)
print ("total nulls:", null_count.sum())


# In[22]:


output_file = DATA_DIR+"test_set_cleaned.csv"
print ("output_file:", output_file)
df_test.to_csv(path_or_buf=output_file)


# In[23]:


#now create new column with average over 5 minute blocks
#first 5-1 minute blocks will be filled using forward five minute average otherwise algo will not generate.


# In[24]:


#create new empty columns for averages.
for colname in colnames[:-1]:
    print ("colname:", colname)
    df_test[colname+'_5'] = np.nan
print ("colnames:", list(df_test))


# In[25]:


df_test.shape


# In[26]:


colnames.index(TIMESTAMP)


# In[27]:


colnames[0:colnames.index(TIMESTAMP)]


# In[28]:


start = time.time()
nrows = 5

for colname in colnames[:colnames.index(TIMESTAMP)]:
    print ("processing colname:", colname)
    for i in range(0, nrows):
        #print ("at i = ", i, "\n", df_test[colname].iloc[i])
        avg_ = df_test[colname].iloc[i:i+nrows].mean()
        #print ("colname:", colname, "type(avg_):", type(avg_), avg_)
        df_test[colname+"_5"].iloc[i] = avg_

end = time.time()
print("time to average first ", nrows, " rows:", end - start)


# In[ ]:


#TODO: check how to speed up next block. not performant.
#possibly lambda function???


# In[ ]:


start = time.time()

for colname in colnames[:colnames.index(TIMESTAMP)]:
    start2 = time.time()

    print ("processing colname:", colname)
    for i in range(nrows, df_test.shape[0]):
        #print ("at i = ", i, "\n", df_test.iloc[i])
        avg_ = df_test[colname].iloc[i-nrows+1:i+1].mean()
        #print ("type(avg_):", type(avg_))
        df_test[colname+"_5"].iloc[i] = avg_
        #if i==100: break  
        if i%1000==0: 
            print ("i:", i, ", elapsed time:", time.time()-start2)
    end2 = time.time()
    print("time to average:", end2 - start2)

end = time.time()
print("time to average first ", nrows, " rows:", end - start)


# In[ ]:


df_test.head(6)


# In[ ]:


df_test.head(110)


# In[ ]:


output_file = DATA_DIR+"test_set_cleaned_norm_5.csv"
print ("output_file:", output_file)
df_test.to_csv(path_or_buf=output_file)

