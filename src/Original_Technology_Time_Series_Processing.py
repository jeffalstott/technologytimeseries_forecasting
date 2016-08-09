
# coding: utf-8

# In[116]:

import pandas as pd
data_directory = '../data/'


# In[136]:

time_series_metadata = pd.DataFrame(columns=['Units', 'Source'])
time_series_metadata.index.name='Name'


# In[137]:

df = pd.read_csv(data_directory+'original/Farmer_Lafond_Data.csv', index_col=0)

for col in df.columns:
    time_series_metadata.ix[col, 'Units'] = df[col].iloc[0]
    time_series_metadata.ix[col, 'Source'] = 'Farmer_Lafond'

df = df.iloc[2:]

for col in df.columns:
    time_series_metadata.ix[col, 'n'] = df[col].notnull().sum()
    time_series_metadata.ix[col, 'Start'] = df[col].dropna().index[0]
    time_series_metadata.ix[col, 'Stop'] = df[col].dropna().index[-1]    


# In[138]:

time_series = df.copy()


# In[154]:

from os import listdir
files = sort(listdir(data_directory+'original/Benson_Magee_Data/'))


# In[159]:

for f in files[::-1]:
    if f.endswith('.xlsx') and 'combo' not in f:
        df = pd.read_excel(data_directory+'original/Benson_Magee_Data/'+f,sheetname='rawdata',index_col=0)
        col = f.split('_v1')[0]
        time_series_metadata.ix[col, 'Units'] = df.columns[0]
        time_series_metadata.ix[col, 'Source'] = 'Benson_Magee'
        time_series_metadata.ix[col, 'n'] = df.dropna().shape[0]
        time_series_metadata.ix[col, 'Start'] = df.dropna().index[0]
        time_series_metadata.ix[col, 'Stop'] = df.dropna().index[-1]
        df.rename(columns={df.columns[0]:col}, inplace=True)
        df.index = df.index.astype('float')
        df = 1/df
        df = df.groupby(level=0).min()
        time_series = time_series.join(df, how='outer')


# In[160]:

time_series = time_series.astype('float')


# In[113]:

time_series.to_csv(data_directory+'time_series.csv')
time_series_metadata.to_csv(data_directory+'time_series_metadata.csv')


# In[103]:

get_ipython().magic('pylab inline')

for c in time_series.columns:
    figure()
    time_series[c].dropna().plot(legend=False)
    yscale('log')
    title(c)


# In[110]:

time_series_metadata[time_series_metadata['Source']=='Farmer_Lafond'].shape


# In[111]:

time_series_metadata[time_series_metadata['Source']=='Benson_Magee'].shape

