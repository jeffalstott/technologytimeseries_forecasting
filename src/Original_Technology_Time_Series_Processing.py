
# coding: utf-8

# In[91]:

import pandas as pd
data_directory = '../data/'


# In[92]:

time_series_metadata = pd.DataFrame(columns=['Units', 'Source'])
time_series_metadata.index.name='Name'


# In[93]:

df = pd.read_csv(data_directory+'original/Farmer_Lafond_Data.csv', index_col=0)

for col in df.columns:
    time_series_metadata.ix[col, 'Units'] = df[col].iloc[0]
    time_series_metadata.ix[col, 'Source'] = 'Farmer_Lafond'
    time_series_metadata.ix[col, 'Type'] = 'Price'


df = df.iloc[2:]

for col in df.columns:
    time_series_metadata.ix[col, 'n'] = df[col].notnull().sum()
    time_series_metadata.ix[col, 'Start'] = df[col].dropna().index[0]
    time_series_metadata.ix[col, 'Stop'] = df[col].dropna().index[-1]    


# In[94]:

time_series = df.copy()


# In[95]:

from os import listdir
from numpy import sort
data_directories = sort(listdir(data_directory+'original/Benson_Magee_Data/'))


# In[96]:

for d in data_directories:
    if d.startswith('.'):
        continue
    files = sort(listdir(data_directory+'original/Benson_Magee_Data/%s/'%d))
    for f in files:
        if f.endswith('.xlsx'):
            df = pd.read_excel(data_directory+'original/Benson_Magee_Data/%s/%s'%(d,f),sheetname='rawdata',index_col=0)
            col = f.split('_v1')[0]
            df = df[df.columns[0]]
            time_series_metadata.ix[col, 'Units'] = df.name
            time_series_metadata.ix[col, 'Source'] = 'Magee_et_al'
            time_series_metadata.ix[col, 'n'] = df.dropna().shape[0]
            time_series_metadata.ix[col, 'Start'] = df.dropna().index[0]
            time_series_metadata.ix[col, 'Stop'] = df.dropna().index[-1]
            df.name = col
            df.index = df.index.astype('float')
            df = 1/df
            df = df.groupby(level=0).min()
            units = time_series_metadata.ix[col, 'Units']
            if "cost" in units or "price" in units or "USD" in units or "$" in units or "dollar" in units:
                time_series_metadata.ix[col, 'Type'] = 'Price'
            else:
                time_series_metadata.ix[col, 'Type'] = 'Performance'
                df = df.cummin().drop_duplicates() #Non-dominated
            time_series = time_series.join(df, how='outer')


# In[97]:

time_series = time_series.astype('float')


# In[100]:

time_series.to_csv(data_directory+'time_series.csv')
time_series_metadata.to_csv(data_directory+'time_series_metadata.csv')


# In[69]:

get_ipython().magic('pylab inline')

for c in time_series.columns:
    figure()
    time_series[c].dropna().plot(legend=False)
    yscale('log')
    title(c)


# In[101]:

time_series_metadata[time_series_metadata['Source']=='Farmer_Lafond'].shape


# In[102]:

time_series_metadata[time_series_metadata['Source']=='Magee_et_al'].shape


# In[107]:

sum(time_series_metadata[time_series_metadata['Source']=='Magee_et_al']['n']>10)

