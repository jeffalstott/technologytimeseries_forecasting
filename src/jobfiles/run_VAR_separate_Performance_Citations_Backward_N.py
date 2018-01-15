#!/home/jeffrey_alstott/anaconda3/bin/python
#PBS -l nodes=1:ppn=4
#PBS -l walltime=20:00:00
#PBS -l mem=10000m
#PBS -N Performance_Citations_Backward_N_VAR_separate

data_type = 'Performance'
target_predictor = 'Citations_Backward_N'
model_type = 'VAR_separate'
    
# coding: utf-8

# In[1]:

training_year = 1990
data_start_year = 1940
# data_type = 'Performance'
# model_type = 'VAR_separate'
# target_predictor = 'meanSPNPcited_1year_before'
# 'Citations_Backward_N', 
# 'Citations_Backward_Age_Mean', 'Citations_Backward_Age_STD', 
# 'meanSPNPcited_1year_before', 'stdSPNPcited_1year_before',
# 'N_Patents
output_directory = '/home/jeffrey_alstott/technoinnovation/technologytimeseries_forecasting/src/'


# Initial setup
# ===

# In[2]:

### Initial setup
from pylab import *
import pandas as pd
import seaborn as sns
sns.set_color_codes()
import pickle


# In[3]:

import sys
sys.path.append('/home/jeffrey_alstott/technoinnovation/')
sys.path.append('/home/jeffrey_alstott/technoinnovation/technologytimeseries_forecasting/src/')
from pystan_time_series import TimeSeriesModel


# In[4]:

### Stan settings and testing functions
n_jobs = 4
n_iterations = 500

def check_div(fit, parameters):
    div = concatenate([s['divergent__'] for s in fit.get_sampler_params(inc_warmup=False)]).astype('bool')

    if sum(div==0):
        print("\x1b[32m\"No divergences\"\x1b[0m")
    else:
        from ndtest import ks2d2s
        divergences = {}
        non_divergences = {}
        for parameter in parameters:
            divergences[parameter] = fit[parameter][div].squeeze()
            non_divergences[parameter] = fit[parameter][~div].squeeze()
            if divergences[parameter].ndim>2:
                N = divergences[parameter].shape[3]
                for n in arange(N):
                    divergences[parameter+'.%i'%n] = divergences[parameter][:,:,n]
                    non_divergences[parameter+'.%i'%n] = non_divergences[parameter][:,:,n]
                del divergences[parameter]
                del non_divergences[parameter]

            any_unevenly_distributed = False
            
            for k1 in divergences.keys():
                for k2 in divergences.keys():
                    if k1==k2:
                        continue

                    x = divergences[k1].ravel()
                    y = divergences[k2].ravel()

                    x_non = non_divergences[k1].ravel()
                    y_non = non_divergences[k2].ravel()

                    p = ks2d2s(x_non, y_non, x, y)
                    if p<.05:
                        any_unevenly_distributed = True
#                         figure()
#                         scatter(x_non, y_non,
#                            alpha=.1, label='Non-Divergent')
#                         scatter(x,y,
#                                alpha=1, label='Divergent')
#                         xlabel(k1)
#                         ylabel(k2)
#                         legend()
#                         title("KS test p=%.2f"%(p))
        if any_unevenly_distributed:
            print("\x1b[31m\"%.2f divergences, which appear to be non-spurious\"\x1b[0m"%(div.mean()))
        else:
            print("\x1b[32m\"%.2f divergences, which appear to be spurious\"\x1b[0m"%(div.mean()))

import stan_utility
def test_model_fit(fit, parameters, max_depth=10):
    stan_utility.check_treedepth(fit,max_depth=max_depth)
    stan_utility.check_energy(fit)
    check_div(fit, parameters)
            
from time import time

def time_series_with_data_for_training_and_testing(Y, training_year, min_observed=3):
    time_series_has_data = (Y[:training_year].notnull().sum(axis=0)>=min_observed) & (Y[training_year+1:].notnull().sum(axis=0)>0)
    time_series_has_data = time_series_has_data[time_series_has_data].index
    return time_series_has_data

def test_prediction(Y_pred, Y_testing):
    inds = where(~isnan(Y_testing))
    inds = zip(*inds)
    from scipy.stats import percentileofscore, gaussian_kde
    d = [i+(gaussian_kde(Y_pred[i]).logpdf(Y_testing[i])[0],
            percentileofscore(Y_pred[i], Y_testing[i])) for i in inds]
    
    predictions_df = pd.DataFrame(columns=['K', 'T', 'D', 'lpdf', 'percentile'],
                                  data = d
                                 )
    return predictions_df

def test_forecasts(Y_prediction, Y_testing):
    forecast_quality = test_prediction(Y_prediction.transpose([1,2,3,0]), Y_testing.values)
    for (label, dimension) in [('K', 0), ('T', 1), ('D', 2)]:
        forecast_quality[label].replace(arange(len(Y_testing.axes[dimension])), 
                                                      Y_testing.axes[dimension],
                                                     inplace=True)
    forecast_quality.set_index(['T', 'K', 'D'], inplace=True)
    forecast_quality.sort_index(inplace=True)
    return forecast_quality


# Empirical Data
# ===

# Performance Data
# ---

# In[5]:

data_directory = '/home/jeffrey_alstott/technoinnovation/technologytimeseries_forecasting/data/'

empirical_time_series = pd.read_csv(data_directory+'time_series.csv',index_col=0)
empirical_time_series.sort_index(axis=1, inplace=True)
empirical_time_series = empirical_time_series.reindex(arange(empirical_time_series.index[0],empirical_time_series.index[-1]+1))
metadata = pd.read_csv(data_directory+'time_series_metadata.csv')

target_tech_names = metadata.loc[(metadata['Domain'].notnull()), 'Name']
time_series_with_domains = empirical_time_series[target_tech_names]

valid_time_series = sum(~time_series_with_domains.loc[1976:].isnull())>3
valid_domains = metadata.set_index('Name').loc[valid_time_series.index[valid_time_series]]['Domain'].unique()

print("Number of valid domains for testing with patent data: %i"%valid_domains.size)


# Patent Data
# ---

# In[6]:

patent_data_directory = '/home/jeffrey_alstott/technoinnovation/patent_centralities/data/'

patents = pd.read_hdf(patent_data_directory+'patents.h5', 'df')
citations = pd.read_hdf(patent_data_directory+'citations.h5', 'df')

citations['Citation_Lag'] = citations['Year_Citing_Patent']-citations['Year_Cited_Patent']
backward_citations = citations.groupby('Citing_Patent')

patents['Citations_Backward_N'] = citations.groupby('Citing_Patent').size()[patents['patent_number']].values
patents['Citations_Backward_Age_Mean'] = citations.groupby('Citing_Patent')['Citation_Lag'].mean()[patents['patent_number']].values
patents['Citations_Backward_Age_STD'] = citations.groupby('Citing_Patent')['Citation_Lag'].std()[patents['patent_number']].values

patent_centralities_z = pd.read_hdf(patent_data_directory+'centralities/summary_statistics.h5', 'empirical_z_scores_USPC')
patent_centralities_z.drop('filing_year', axis=1, inplace=True)
patents = patents.merge(patent_centralities_z, on='patent_number')

patents_percentile_by_year = patents.copy()
for col in patents.columns:
    if col in ['filing_year', 'patent_number', 'Class']:
        continue
    patents_percentile_by_year[col] = patents.groupby('filing_year')[col].rank(pct=True)

patents_percentile_by_year.set_index('patent_number', inplace=True)

n_patents_by_year = patents.groupby('filing_year').size()


# In[7]:

patent_domains = pd.read_csv(data_directory+'PATENT_SET_DOMAINS.csv', index_col=0)
def floatify(x):
    from numpy import nan
    try:
        return float(x)
    except ValueError:
        return nan
patent_domains['patent_id'] = patent_domains['patent_id'].apply(floatify)
patent_domains = patent_domains.dropna()
domains = patent_domains['Domain'].unique() 


# In[8]:

candidate_predictors = ['Citations_Backward_N',
                        'Citations_Backward_Age_Mean',
                        'Citations_Backward_Age_STD',
                        'meanSPNPcited_1year_before',
                        'stdSPNPcited_1year_before',
                       ]

first_year_for_predictors = 1976
for col in candidate_predictors+['filing_year']:
    patent_domains[col] = patents_percentile_by_year.loc[patent_domains['patent_id'], col].values
predictors_by_domain = patent_domains.groupby(['Domain', 'filing_year'])[candidate_predictors].mean()
predictors_by_domain['N_Patents'] = patent_domains.groupby(['Domain', 'filing_year']).size()

predictors_by_domain = predictors_by_domain.reindex([(d,y) for d in domains for y in arange(first_year_for_predictors, 
                                                                                                  patent_domains['filing_year'].max()+1)])

predictors_by_domain['N_Patents'] = predictors_by_domain['N_Patents'].fillna(0).values/n_patents_by_year.loc[predictors_by_domain.reset_index()['filing_year']].values


# In[9]:

patent_time_series = pd.Panel(items=empirical_time_series.columns,
                      major_axis=empirical_time_series.index,
                      minor_axis=predictors_by_domain.columns,#+['Performance'],
                     )

for predictor in predictors_by_domain.columns:
    print(predictor)
    for col in empirical_time_series.columns:
        if metadata.set_index('Name').notnull().loc[col, 'Domain']:
            patent_time_series.loc[col,:, predictor] = predictors_by_domain.loc[metadata.set_index('Name').loc[col, 
                                                                                                   'Domain']][predictor]
# predictors.fillna(0, inplace=True)
# Y_var.loc[:,:, 'Performance'] = Y

# combined_df = Y.stack(dropna=False).swaplevel()
# combined_df.name = 'Performance'
# combined_df = pd.DataFrame(combined_df)

has_patent_data = patent_time_series.notnull().any().iloc[0]
has_patent_data = has_patent_data[has_patent_data].index.values
# Y_var = Y_var.loc[has_patent_data]


# Build Predictive Models
# ===

# In[10]:

target_tech_names = metadata.loc[metadata['Type']==data_type, 'Name']
Y = log(empirical_time_series[target_tech_names]).loc[data_start_year:]
time_series_has_data = time_series_with_data_for_training_and_testing(Y, training_year)
Y = Y[time_series_has_data]
Y = pd.Panel({data_type:Y}).transpose([2,1,0])
Y_training = Y.copy()
Y_testing = Y.copy()
Y_training.loc[:,training_year+1:] = nan
Y_testing.loc[:, :training_year] = nan

# Y_testing = Y_testing[time_series_has_data[:10]]
# Y_training = Y_training[time_series_has_data[:10]]


# In[29]:

def train_model_types(time_series, model_types, training_year=1990, use_partial_pooling=True):
    models = {}
    max_depth = 15
    for model_name in model_types.keys():
        print(model_name+'\n============================')
        start_time = time()
        model = TimeSeriesModel(Y=time_series.values, use_partial_pooling=use_partial_pooling, **model_types[model_name])
        model.sampling(n_jobs=n_jobs, iter=n_iterations, control={'max_treedepth':max_depth})
        test_model_fit(model.fit, ['mu', 'sigma'], max_depth=max_depth)
        models[model_name] = model.fit['Y_latent']
        print("Fitting took %.2f minutes"%((time()-start_time)/60))
    return models

def train_model_types_separate(time_series, model_types, training_year=1990):
    models = {}
    
    for model_name in model_types.keys():
        print(model_name+'\n============================')
        start_time = time()
        separate_models_forecast = []
        for c in time_series.items:
            print(c)
            model = TimeSeriesModel(Y=expand_dims(time_series[c],0), **model_types[model_name])
            max_depth = 15
            model.sampling(n_jobs=n_jobs, iter=n_iterations, control={'max_treedepth':max_depth})
            test_model_fit(model.fit, ['mu', 'sigma'], max_depth=max_depth)
            separate_models_forecast.append(model.fit['Y_latent'])
            print("Fitting has taken %.2f minutes"%((time()-start_time)/60))
        separate_models_forecast = concatenate(separate_models_forecast, axis=1)
        models[model_name] = separate_models_forecast
        print("Fitting took %.2f minutes"%((time()-start_time)/60))

    return models


# In[12]:

if data_type=='Performance':
    monotonic = [1]
else:
    monotonic = None

model_types = {
                'ARIMA(0,0)': {'p':0,'q':0, 'difference': [1], 'monotonic': monotonic},
               'ARIMA(0,1)': {'p':0, 'q':1, 'difference': [1], 'monotonic': monotonic},
               'ARIMA(1,0)': {'p':1,'q':0, 'difference': [1], 'monotonic': monotonic},
               'ARIMA(1,1)': {'p':1,'q':1, 'difference': [1], 'monotonic': monotonic},
               'ARIMA([1,5],0)': {'p':[1,5],'q':0, 'difference': [1], 'monotonic': monotonic},
               'ARIMA([1,5],1)': {'p':[1,5],'q':1, 'difference': [1], 'monotonic': monotonic},
               'ARIMA([1,5],[1,5])': {'p':[1,5],'q':[1,5], 'difference': [1], 'monotonic': monotonic},
              }


# In[13]:

if model_type=='separate':
    print("Modeling separate")
# try:
#     separate_models_forecasts = pickle.load(open('separate_models_forecasts_%s'%data_type, 'rb'))
# except FileNotFoundError:
    separate_models_forecasts = train_model_types_separate(Y_training, model_types)
    separate_models_forecasts['training'] = Y_training
    separate_models_forecasts['testing'] = Y_testing
    pickle.dump(separate_models_forecasts, open(output_directory+'separate_models_forecasts_%s'%data_type, 'wb'))


# In[14]:

if model_type=='pooled':
    print("Modeling pooled")
# try:
#     pooled_models_forecasts = pickle.load(open('pooled_models_forecasts_%s'%data_type, 'rb'))
# except FileNotFoundError:
    pooled_models_forecasts = train_model_types(Y_training, model_types)
    pooled_models_forecasts['training'] = Y_training
    pooled_models_forecasts['testing'] = Y_testing
    pickle.dump(pooled_models_forecasts, open(output_directory+'pooled_models_forecasts_%s'%data_type, 'wb'))


# VAR model
# ===
# 

# In[15]:

data_start_year = 1975
Y = log(empirical_time_series[target_tech_names]).loc[data_start_year:]
time_series_has_data = time_series_with_data_for_training_and_testing(Y, training_year)
Y = Y[time_series_has_data]
Y = pd.Panel({data_type:Y}).transpose([2,1,0])

from scipy.special import logit
Y_var = patent_time_series.loc[time_series_has_data, data_start_year:].copy()
Y_var.loc[:,:,:] = logit(Y_var).values
has_patent_data = Y_var.notnull().any().iloc[0]
has_patent_data = has_patent_data[has_patent_data].index.values
Y_var.loc[:,:, data_type] = Y.loc[:,:,data_type]
Y_var = Y_var.loc[has_patent_data]

Y_var = Y_var.loc[:,:,[target_predictor, data_type]]


Y_var_training = Y_var.copy()
Y_var_testing = Y_var.copy()
Y_var_training.loc[:,training_year+1:] = nan
Y_var_testing.loc[:,:training_year] = nan


# In[21]:

if data_type=='Performance':
    monotonic = [Y_var_training.shape[-1]-1]
else:
    monotonic = None
model_types = {
#                'ARIMA(0,0)': {'p':0,'q':0, 'difference': ones(Y_var.shape[-1]).astype('int')},
               'ARIMA(1,0)': {'p':1,'q':0, 'difference': ones(Y_var_training.shape[-1]).astype('int'), 'monotonic':monotonic},
               'ARIMA(1,1)': {'p':1,'q':1, 'difference': ones(Y_var.shape[-1]).astype('int'), 'monotonic':monotonic},
               'ARIMA([1,5],0)': {'p':[1,5],'q':0, 'difference': ones(Y_var.shape[-1]).astype('int'), 'monotonic':monotonic},
               'ARIMA([1,5],1)': {'p':[1,5],'q':1, 'difference': ones(Y_var.shape[-1]).astype('int'), 'monotonic':monotonic},
               'ARIMA([1,5],[1,5])': {'p':[1,5],'q':[1,5], 'difference': ones(Y_var.shape[-1]).astype('int'), 'monotonic':monotonic},
              }


# In[ ]:

if model_type == 'VAR_separate':
    print("Modeling VAR separate")
    file_name = output_directory+'VAR_%s_separate_models_forecasts_%s'%(target_predictor, data_type)
    VAR_models_forecasts = train_model_types_separate(Y_var_training, model_types)
    VAR_models_forecasts['training'] = Y_var_training
    VAR_models_forecasts['testing'] = Y_var_testing
    pickle.dump(VAR_models_forecasts, open(file_name, 'wb'))


# In[ ]:

if model_type == 'VAR_pooled':
    print("Modeling VAR pooled")
    # try:
    #     VAR_models_forecasts = pickle.load(open('VAR_models_forecasts_%s'%data_type, 'rb'))
    # except FileNotFoundError:
    VAR_models_forecasts = train_model_types(Y_var_training, model_types)
    VAR_models_forecasts['training'] = Y_var_training
    VAR_models_forecasts['testing'] = Y_var_testing
    pickle.dump(VAR_models_forecasts, open('VAR_pooled_models_forecasts_%s'%data_type, 'wb'))

