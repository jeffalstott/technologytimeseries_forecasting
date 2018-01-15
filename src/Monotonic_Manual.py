
# coding: utf-8

# In[ ]:

from pystan import StanModel
n_jobs = 4
import pandas as pd
import seaborn as sns
sns.set_color_codes()
import pickle
get_ipython().magic('pylab inline')
models = pickle.load(open('model.pkl', 'rb'))


# In[ ]:

def test_model_inference(model_name, Y=None, predictors=None, generated_data='data_latent', models=models, 
                         generator_iter=50, inference_iter=1000):
    
    if Y is None:
        Y = pd.DataFrame(rand(100,5))
    if predictors is None:
        stan_data = models[model_name]['stan_data_creator'](Y, run_inference=False)
    else:
        stan_data = models[model_name]['stan_data_creator'](Y, predictors,run_inference=False)
    stan_data = {**stan_data, 
                 **models[model_name]['parameter_priors']} 

    generated_example = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=generator_iter)

    sample = 20
    generated_parameters = {}
    for parameter in models[model_name]['model_parameters']:
        generated_parameters[parameter] = generated_example[parameter][sample]

    generated_data = pd.DataFrame(generated_example[generated_data][sample])

    if predictors is None:
        stan_data = models[model_name]['stan_data_creator'](generated_data, run_inference=True)
    else:
        stan_data = models[model_name]['stan_data_creator'](generated_data, predictors,run_inference=True)
    stan_data = {**stan_data, 
                 **models[model_name]['parameter_priors']}  
    
    model_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=inference_iter)

    true_parameters_inferred_scores = {}
    true_parameters_inferred_score_within_95CI = 0
    n_parameters = 0
    from scipy.stats import percentileofscore
    
    for parameter in models[model_name]['model_parameters']:
        parameter_samples = model_fit[parameter]
        if parameter_samples.ndim>2:
            parameter_samples = parameter_samples.reshape(parameter_samples.shape[0], 
                                                          prod(parameter_samples.shape[1:]))
        true_parameters_inferred_scores[parameter] = array(list(map(percentileofscore, 
                                                             parameter_samples.T, 
                                                             generated_parameters[parameter].ravel())))
        true_parameters_inferred_score_within_95CI += sum((true_parameters_inferred_scores[parameter]>2.5) & 
                                                          (true_parameters_inferred_scores[parameter]<97.5)
                                                         )
        n_parameters += true_parameters_inferred_scores[parameter].size
    return true_parameters_inferred_score_within_95CI/n_parameters#, true_parameters_inferred_score_within_95CI

from pystan.misc import _summary, _array_to_table
def _print_stanfit(fit, pars=None, probs=(0.025, 0.25, 0.5, 0.75, 0.975), digits_summary=2):
        if fit.mode == 1:
            return "Stan model '{}' is of mode 'test_grad';\n"                   "sampling is not conducted.".format(fit.model_name)
        elif fit.mode == 2:
            return "Stan model '{}' does not contain samples.".format(fit.model_name)
        if pars is None:
            pars = fit.sim['pars_oi']
            fnames = fit.sim['fnames_oi']

        n_kept = [s - w for s, w in zip(fit.sim['n_save'], fit.sim['warmup2'])]
        header = ""#Inference for Stan model: {}.\n".format(fit.model_name)
        header += "{} chains, each with iter={}; warmup={}; thin={}; \n"
        header = header.format(fit.sim['chains'], fit.sim['iter'], fit.sim['warmup'],
                               fit.sim['thin'], sum(n_kept))
        header += "post-warmup draws per chain={}, total post-warmup draws={}.\n\n"
        header = header.format(n_kept[0], sum(n_kept))
        footer = "\n\nSamples were drawn using {} at {}.\n"#             "For each parameter, n_eff is a crude measure of effective sample size,\n"\
#             "and Rhat is the potential scale reduction factor on split chains (at \n"\
#             "convergence, Rhat=1)."
        sampler = fit.sim['samples'][0]['args']['sampler_t']
        date = fit.date.strftime('%c')  # %c is locale's representation
        footer = footer.format(sampler, date)
        s = _summary(fit, pars, probs)
        body = _array_to_table(s['summary'], s['summary_rownames'],
                               s['summary_colnames'], digits_summary)
        return header + body + footer

def plot_time_series_inference(model_fit, var='data_latent', x=None,
                               ax=None, ind=0, **kwargs):
    from scipy.stats import scoreatpercentile
    ci_thresholds = [2.5, 25, 75, 97.5]
    if len(model_fit[var].shape)<3:
        data = model_fit[var]
    else:
        data = model_fit[var][:,:,ind]
    CIs = scoreatpercentile(data, ci_thresholds, axis=0)
    CIs = pd.DataFrame(data=CIs.T, columns=ci_thresholds)
    if ax is None:
        ax=gca()
    if x is None:
        x = arange(data.shape[1])
    ax.fill_between(x, CIs[2.5], CIs[97.5],alpha=.5, **kwargs)
    ax.fill_between(x, CIs[25], CIs[75], **kwargs)


# In[3]:

data_directory = '../data/'

empirical_data = pd.read_csv(data_directory+'time_series.csv',index_col=0)
empirical_data = empirical_data.reindex(arange(empirical_data.index[0],empirical_data.index[-1]+1))
metadata = pd.read_csv(data_directory+'time_series_metadata.csv')

target_tech_names = metadata.loc[(metadata['Source']=='Farmer_Lafond')*(metadata['Type']=='Price'), 'Name']
empirical_time_series = log(empirical_data[target_tech_names])

# valid_time_series = sum(~empirical_time_series.loc[1976:].isnull())>3
# valid_domains = metadata.set_index('Name').loc[valid_time_series.index[valid_time_series]]['Domain'].unique()

# print("Number of valid domains: %i"%valid_domains.size)


# In[56]:

model_name = 'Y~ARMA'
models[model_name] = {}


models[model_name]['code'] = """

data {

    int T; // number of time steps
    int K; // Number of time series
    int P; // Number of predictors
    int L; // Number of lags for ARMA element
    
    matrix[T, K] Y; // data to model
    matrix[T, P] predictors[K]; // predictors
    
    int first_observation[K]; // index of first observation in each time series
    int last_observation[K]; // index of last observation in each time series
    
    int n_missing_observations_before_first_and_last; // number of missing observations before and after the end of the time series
    int n_missing_updates_between_first_and_last; // number of missing updates (steps between each observation) with the time series
    
    int run_inference;
    
    // priors
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real phi_prior_location;
    real phi_prior_scale;
    
    real theta_prior_location;
    real theta_prior_scale;
    
    //real beta_prior_location;
    //real beta_prior_scale;
}

parameters {
    vector[K] mu;
    vector[K] sigma;
    matrix[K,L] phi;
    //matrix[K,L] theta;
    //matrix[K,P] beta;
    
    vector[n_missing_observations_before_first_and_last] free_latent_parameters;
    vector[n_missing_updates_between_first_and_last] restricted_latent_parameters;

}
transformed parameters {
    matrix[T,K] Y_latent;
    
    
    // Fill the latent data before and after the observed data with completely unrestricted parameters
    {
    int free_param_counter;
    free_param_counter = 1;
    
    for (k in 1:K){
    
        if (first_observation[k]>1){
        Y_latent[1:first_observation[k]-1, k] = 
        free_latent_parameters[free_param_counter:free_param_counter+first_observation[k]-1];
        free_param_counter = free_param_counter + first_observation[k]-1;
        }
        
        if (last_observation[k]<T){
        Y_latent[last_observation[k]+1:T, k] = 
        free_latent_parameters[free_param_counter:free_param_counter+T-last_observation[k]];
        free_param_counter = free_param_counter + T-last_observation[k];
        }
    }
    }
    
    
    // Fill the latent data within the observed data with either data values or restricted parameters
    {
    int restricted_param_counter;
    int gap_width;
    real previous_value;
    int previous_value_index;
    
    restricted_param_counter = 1;
    
    
    for (k in 1:K){
            previous_value = Y[first_observation[k],k];
            Y_latent[first_observation[k],k] = Y[first_observation[k],k];
            previous_value_index = first_observation[k];
        for (t in first_observation[k]+1:last_observation[k]){
            if (Y[t,k]>-999){
                gap_width = t-previous_value_index-1;
                if (gap_width>0){
                    // These are the unobserved UPDATES between observed time steps. 
                    // I.e. If Y_3 and Y_1 are observed, by Y_2 is not, these are (Y_3 - Y_2) and (Y_2-Y_1)
                    // We will say that these updates have to sum up to the observed difference between Y_3 and Y_1.
                    // The unobserved time steps then have values that are the cumulative sum of these updates.

                    
                    Y_latent[previous_value_index+1:t, k] = 
                    cumulative_sum(
                     restricted_latent_parameters[restricted_param_counter:(restricted_param_counter+gap_width+1)]
                     / sum(restricted_latent_parameters[restricted_param_counter:restricted_param_counter+gap_width+1])
                     * (Y[t,k] - previous_value)
                     ) + previous_value;
                    
                    
                    // Don't need to include the last update in this sum, since we can explicitly grab the level
                    // that we get to form the data itself.
                    //data_latent[previous_value_index+1:t-1, k] = 
                    //cumsum(restricted_latent_parameters[restricted_param_counter:restricted_param_counter+gap_width])
                    //+ previous_value;
                    
                }
                Y_latent[t,k] = Y[t,k];
                previous_value = Y[t,k];
                previous_value_index = t;
            } 
        }
    }
    }
}

model {
    matrix[T,K] err;
    matrix[T,K] nu;
    
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    //for (i in 1:rows(beta)){
    //    beta[i] ~ normal(beta_prior_location, beta_prior_scale);
    //}
    
    phi[:,1] ~ normal(1, phi_prior_scale); //prior is centered around random walk
    if (L>1){
    for (i in 2:L){
        phi[:,i] ~ normal(phi_prior_location, phi_prior_scale);
    }
    }
    
    //for (i in 1:rows(theta)){
    //    theta[i] ~ normal(theta_prior_location, theta_prior_scale);
    //}
    
    for (k in 1:K){
        err[:,k] ~ normal(0, sigma[k]);
    }
    
    if(run_inference==1){
        for (k in 1:K) {
            
            for (t in (L+1):T){
                nu[t,k] = mu[k] + phi[k]*Y_latent[t-L:t-1, k];// + theta[k]*err[t-L:t-1, k]; //+ exp(beta[k]*predictors[k][t])
                err[t,k] = Y_latent[t,k] - nu[t,k];
            }
            
            nu[1,k] = mu[k] + phi[k,1]*mu[k]; //+ exp(beta[k]*predictors[k][1])
            err[1,k] = Y_latent[1,k] - nu[1,k];
            
            if (L>1){
            for (t in 2:L){
                nu[t,k] = mu[k] + phi[k,1:t-1]*Y_latent[1:t-1, k];// + theta[k, 1:t-1]*err[1:t-1, k]; //+ exp(beta[k]*predictors[k][t])
                err[t,k] = Y_latent[t,k] - nu[t,k];
            }
            }
        }
    }
}
"""

models[model_name]['stan_model'] = StanModel(model_code=models[model_name]['code'])

models[model_name]['parameter_priors'] = {
    'mu_prior_location': 0,
    'mu_prior_scale': 1,
    'sigma_prior_location': 0,
    'sigma_prior_scale': 1,
    'phi_prior_location': 0,
    'phi_prior_scale': 1,
    'theta_prior_location': 0,
    'theta_prior_scale': 1,
#     'beta_prior_location': 0,
#     'beta_prior_scale': 1,
    }

models[model_name]['model_parameters'] = unique([i.split('_prior')[0] for i in models[model_name]['parameter_priors'].keys()])

def stan_data_creator(Y, predictors=None, L=3, run_inference=True):
    Y = Y.copy()
    T = Y.shape[0]
    K = Y.shape[1]
    Y.index = range(T)
    Y.columns = range(K)
    first_observation = Y.apply(lambda x: x.first_valid_index())
    last_observation = Y.apply(lambda x: x.last_valid_index())
    n_missing_observations_before_first_and_last = sum(first_observation)+sum((T-1)-last_observation)
    n_missing_updates_between_first_and_last = sum([Y.loc[first_observation[k]:last_observation[k], k].diff().isnull()[1:].sum() for k in range(K)])
    
    if predictors is None:
        predictors = ones((K,T,0))
    stan_data = {'Y':Y.fillna(-999),
                 'T': T,
                 'K': K,
                 'L': L,
                 'first_observation': first_observation.astype('int')+1,
                 'last_observation': last_observation.astype('int')+1,
                 'n_missing_observations_before_first_and_last': n_missing_observations_before_first_and_last,
                 'n_missing_updates_between_first_and_last': n_missing_updates_between_first_and_last,
                 'P': predictors.shape[-1],
                 'predictors': predictors,
                 'run_inference': int(run_inference),
                }
    return stan_data

models[model_name]['stan_data_creator'] = stan_data_creator


# In[63]:

model_name = 'Y~AR'
models[model_name] = {}


models[model_name]['code'] = """

data {

    int T; // number of time steps
    int K; // Number of time series
    int P; // Number of lags for AR element
    
    matrix[T, K] Y; // data to model
    
    int first_observation[K]; // index of first observation in each time series
    int last_observation[K]; // index of last observation in each time series
    
    int n_missing_observations_before_first_and_last; // number of missing observations before and after the end of the time series
    int n_missing_updates_between_first_and_last; // number of missing updates (steps between each observation) with the time series
    
    int run_inference;
    
    // priors
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real phi_prior_location;
    real phi_prior_scale;
    
    //real theta_prior_location;
    //real theta_prior_scale;

}

parameters {
    vector[K] mu;
    vector[K] sigma;
    matrix[K,P] phi;
    
    vector[n_missing_observations_before_first_and_last] free_latent_parameters;
    vector[n_missing_updates_between_first_and_last] restricted_latent_parameters;

}
transformed parameters {
    matrix[T,K] Y_latent;
    
    
    // Fill the latent data before and after the observed data with completely unrestricted parameters
    {
    int free_param_counter;
    free_param_counter = 1;
    
    for (k in 1:K){
    
        if (first_observation[k]>1){
        Y_latent[1:first_observation[k]-1, k] = 
        free_latent_parameters[free_param_counter:free_param_counter+first_observation[k]-1];
        free_param_counter = free_param_counter + first_observation[k]-1;
        }
        
        if (last_observation[k]<T){
        Y_latent[last_observation[k]+1:T, k] = 
        free_latent_parameters[free_param_counter:free_param_counter+T-last_observation[k]];
        free_param_counter = free_param_counter + T-last_observation[k];
        }
    }
    }
    
    
    // Fill the latent data within the observed data with either data values or restricted parameters
    {
    int restricted_param_counter;
    int gap_width;
    real previous_value;
    int previous_value_index;
    
    restricted_param_counter = 1;
    
    
    for (k in 1:K){
            previous_value = Y[first_observation[k],k];
            Y_latent[first_observation[k],k] = Y[first_observation[k],k];
            previous_value_index = first_observation[k];
        for (t in first_observation[k]+1:last_observation[k]){
            if (Y[t,k]>-999){
                gap_width = t-previous_value_index-1;
                if (gap_width>0){
                    // These are the unobserved UPDATES between observed time steps. 
                    // I.e. If Y_3 and Y_1 are observed, by Y_2 is not, these are (Y_3 - Y_2) and (Y_2-Y_1)
                    // We will say that these updates have to sum up to the observed difference between Y_3 and Y_1.
                    // The unobserved time steps then have values that are the cumulative sum of these updates.

                    
                    Y_latent[previous_value_index+1:t, k] = 
                    cumulative_sum(
                     restricted_latent_parameters[restricted_param_counter:(restricted_param_counter+gap_width+1)]
                     / sum(restricted_latent_parameters[restricted_param_counter:restricted_param_counter+gap_width+1])
                     * (Y[t,k] - previous_value)
                     ) + previous_value;
                    
                    
                    // Don't need to include the last update in this sum, since we can explicitly grab the level
                    // that we get to form the data itself.
                    //data_latent[previous_value_index+1:t-1, k] = 
                    //cumsum(restricted_latent_parameters[restricted_param_counter:restricted_param_counter+gap_width])
                    //+ previous_value;
                    
                }
                Y_latent[t,k] = Y[t,k];
                previous_value = Y[t,k];
                previous_value_index = t;
            } 
        }
    }
    }
}

model {
    matrix[T,K] err;
    matrix[T,K] nu;
    
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    if (P>0){
    phi[:,1] ~ normal(1, phi_prior_scale); //prior is centered around random walk
    }
    if (P>1){
    for (p in 2:P){
        phi[:,p] ~ normal(phi_prior_location, phi_prior_scale);
    }
    }
    
    
    for (k in 1:K) {
        nu[:,k] = rep_vector(mu[k], T);
        if (P>0){
            for (t in P+1:T){
                nu[t,k] = nu[t,k] + phi[k]*Y_latent[t-P:t-1,k];
                }
            }
    }
    
    err = Y_latent - nu;
    
    for (k in 1:K){
        err[P+1:T,k] ~ normal(0, sigma[k]);
    }
}
"""

models[model_name]['stan_model'] = StanModel(model_code=models[model_name]['code'])

models[model_name]['parameter_priors'] = {
    'mu_prior_location': 0,
    'mu_prior_scale': 1,
    'sigma_prior_location': 0,
    'sigma_prior_scale': 1,
    'phi_prior_location': 0,
    'phi_prior_scale': 1,
#     'theta_prior_location': 0,
#     'theta_prior_scale': 1,
#     'beta_prior_location': 0,
#     'beta_prior_scale': 1,
    }

models[model_name]['model_parameters'] = unique([i.split('_prior')[0] for i in models[model_name]['parameter_priors'].keys()])

def stan_data_creator(Y, predictors=None, p=1, run_inference=True):
    Y = Y.copy()
    T = Y.shape[0]
    K = Y.shape[1]
    Y.index = range(T)
    Y.columns = range(K)
    first_observation = Y.apply(lambda x: x.first_valid_index())
    last_observation = Y.apply(lambda x: x.last_valid_index())
    n_missing_observations_before_first_and_last = sum(first_observation)+sum((T-1)-last_observation)
    n_missing_updates_between_first_and_last = sum([Y.loc[first_observation[k]:last_observation[k], k].diff().isnull()[1:].sum() for k in range(K)])
    
    stan_data = {'Y':Y.fillna(-999),
                 'T': T,
                 'K': K,
                 'P': p,
                 'first_observation': first_observation.astype('int')+1,
                 'last_observation': last_observation.astype('int')+1,
                 'n_missing_observations_before_first_and_last': n_missing_observations_before_first_and_last,
                 'n_missing_updates_between_first_and_last': n_missing_updates_between_first_and_last,
                 'run_inference': int(run_inference),
                }
    return stan_data

models[model_name]['stan_data_creator'] = stan_data_creator


# In[70]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~AR'\nY = pd.DataFrame(rand(100,3))\n# Y.iloc[0] = nan\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=0), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nmodel_fit")


# In[66]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~AR'\nY = pd.DataFrame(cumsum(rand(100,3)*3, axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=1), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nmodel_fit")


# In[40]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~AR'\nY = pd.DataFrame(cumsum(cumsum(rand(100,3)*3, axis=0), axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=2), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)")


# In[37]:

model_name = 'Y~ARMA'
models[model_name] = {}


models[model_name]['code'] = """

data {

    int T; // number of time steps
    int K; // Number of time series
    int<lower=0,upper=T-1> P; // Number of lags for AR element
    int<lower=0,upper=T-1> Q; // Number of lags for MA element
    
    matrix[T, K] Y; // data to model
    
    int first_observation[K]; // index of first observation in each time series
    int last_observation[K]; // index of last observation in each time series
    
    int n_missing_observations_before_first_and_last; // number of missing observations before and after the end of the time series
    int n_missing_updates_between_first_and_last; // number of missing updates (steps between each observation) with the time series
    
    int run_inference;
    
    // priors
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real phi_prior_location;
    real phi_prior_scale;
    
    real theta_prior_location;
    real theta_prior_scale;

}

parameters {
    vector[K] mu;
    vector[K] sigma;
    matrix[K,P] phi;
    matrix<lower = -1, upper = 1>[K,Q] theta;
    
    vector[n_missing_observations_before_first_and_last] free_latent_parameters;
    vector[n_missing_updates_between_first_and_last] restricted_latent_parameters;

}
transformed parameters {
    matrix[T,K] Y_latent;
    
    
    // Fill the latent data before and after the observed data with completely unrestricted parameters
    {
    int free_param_counter;
    free_param_counter = 1;
    
    for (k in 1:K){
    
        if (first_observation[k]>1){
        Y_latent[1:first_observation[k]-1, k] = 
        free_latent_parameters[free_param_counter:free_param_counter+first_observation[k]-1];
        free_param_counter = free_param_counter + first_observation[k]-1;
        }
        
        if (last_observation[k]<T){
        Y_latent[last_observation[k]+1:T, k] = 
        free_latent_parameters[free_param_counter:free_param_counter+T-last_observation[k]];
        free_param_counter = free_param_counter + T-last_observation[k];
        }
    }
    }
    
    
    // Fill the latent data within the observed data with either data values or restricted parameters
    {
    int restricted_param_counter;
    int gap_width;
    real previous_value;
    int previous_value_index;
    
    restricted_param_counter = 1;
    
    
    for (k in 1:K){
            previous_value = Y[first_observation[k],k];
            Y_latent[first_observation[k],k] = Y[first_observation[k],k];
            previous_value_index = first_observation[k];
        for (t in first_observation[k]+1:last_observation[k]){
            if (Y[t,k]>-999){
                gap_width = t-previous_value_index-1;
                if (gap_width>0){
                    // These are the unobserved UPDATES between observed time steps. 
                    // I.e. If Y_3 and Y_1 are observed, by Y_2 is not, these are (Y_3 - Y_2) and (Y_2-Y_1)
                    // We will say that these updates have to sum up to the observed difference between Y_3 and Y_1.
                    // The unobserved time steps then have values that are the cumulative sum of these updates.

                    
                    Y_latent[previous_value_index+1:t, k] = 
                    cumulative_sum(
                     restricted_latent_parameters[restricted_param_counter:(restricted_param_counter+gap_width+1)]
                     / sum(restricted_latent_parameters[restricted_param_counter:restricted_param_counter+gap_width+1])
                     * (Y[t,k] - previous_value)
                     ) + previous_value;
                    
                    
                    // Don't need to include the last update in this sum, since we can explicitly grab the level
                    // that we get to form the data itself.
                    //data_latent[previous_value_index+1:t-1, k] = 
                    //cumsum(restricted_latent_parameters[restricted_param_counter:restricted_param_counter+gap_width])
                    //+ previous_value;
                    
                }
                Y_latent[t,k] = Y[t,k];
                previous_value = Y[t,k];
                previous_value_index = t;
            } 
        }
    }
    }
}

model {
    matrix[T,K] err;
    matrix[T,K] nu;
    
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    if (P>0){
    phi[:,1] ~ normal(1, phi_prior_scale); //prior is centered around random walk
    }
    if (P>1){
    for (p in 2:P){
        phi[:,p] ~ normal(phi_prior_location, phi_prior_scale);
    }
    }
    
    
    for (k in 1:K) {
        nu[:,k] = rep_vector(mu[k], T);
        if (P>0){
            for (t in P+1:T){
                nu[t,k] = nu[t,k] + phi[k]*Y_latent[t-P:t-1,k];
                }
            }
        if (Q==0){
        err[:,k] = Y_latent[:,k] - nu[:,k];
        }
        else{
            //Need to sort out initial cases here.
            nu[1,k] = mu[k] + phi[k,1]*mu[k]; 
            err[1,k] = Y_latent[1,k] - nu[1,k];
            if (Q>1){
                for (t in 2:Q){
                    nu[t,k] = nu[t,k] + phi[k,1:t-1]*Y_latent[1:t-1, k] + theta[k,1:t-1]*err[1:t-1, k];
                    err[t,k] = Y_latent[t,k] - nu[t,k];
                }
            }
            for (t in Q+1:T){
                nu[t,k] = nu[t,k] + theta[k]*err[t-Q:t-1,k]; // Damn. This adding thetas effect on top of phis effect won't work. They have to be calculated together. Or does it? It depends on whether the phis are working on lagged Y_latent or lagged nu. They're working on lagged Y_latent, so we should be safe, right?  
                err[t,k] = Y_latent[t,k] - nu[t,k];
                }
        }
        
    }
        
    for (k in 1:K){
        err[max(P+1,Q+1):T,k] ~ normal(0, sigma[k]);
    }
}
"""

models[model_name]['stan_model'] = StanModel(model_code=models[model_name]['code'])

models[model_name]['parameter_priors'] = {
    'mu_prior_location': 0,
    'mu_prior_scale': 1,
    'sigma_prior_location': 0,
    'sigma_prior_scale': 1,
    'phi_prior_location': 0,
    'phi_prior_scale': 1,
    'theta_prior_location': 0,
    'theta_prior_scale': 1,
#     'beta_prior_location': 0,
#     'beta_prior_scale': 1,
    }

models[model_name]['model_parameters'] = unique([i.split('_prior')[0] for i in models[model_name]['parameter_priors'].keys()])

def stan_data_creator(Y, predictors=None, p=1, q=1, run_inference=True):
    Y = Y.copy()
    T = Y.shape[0]
    K = Y.shape[1]
    Y.index = range(T)
    Y.columns = range(K)
    first_observation = Y.apply(lambda x: x.first_valid_index())
    last_observation = Y.apply(lambda x: x.last_valid_index())
    n_missing_observations_before_first_and_last = sum(first_observation)+sum((T-1)-last_observation)
    n_missing_updates_between_first_and_last = sum([Y.loc[first_observation[k]:last_observation[k], k].diff().isnull()[1:].sum() for k in range(K)])
    
    stan_data = {'Y':Y.fillna(-999),
                 'T': T,
                 'K': K,
                 'P': p,
                 'Q': q,
                 'first_observation': first_observation.astype('int')+1,
                 'last_observation': last_observation.astype('int')+1,
                 'n_missing_observations_before_first_and_last': n_missing_observations_before_first_and_last,
                 'n_missing_updates_between_first_and_last': n_missing_updates_between_first_and_last,
                 'run_inference': int(run_inference),
                }
    return stan_data

models[model_name]['stan_data_creator'] = stan_data_creator


# In[ ]:

nu[1,k] = mu[k] + phi[k,1]*mu[k];
        err[1,k] = Y_latent[1,k] - nu[1,k];

        if (P>1){
        for (t in 2:P){
            nu[t,k] = mu[k] + dot_product(phi[k,1:t-1],Y_latent[1:t-1, k]);
            err[t,k] = Y_latent[t,k] - nu[t,k];
        }
        }

        for (t in (P+1):T){
            y[2:(N - 1)] ~ normal(alpha + beta * y[1:(N - 1)], sigma);
            nu[t,k] = mu[k] + dot_product(phi[k],Y_latent[t-P:t-1, k]);
            err[t,k] = Y_latent[t,k] - nu[t,k];
        }
        }


# In[38]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(rand(100,3)*3)\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=0,q=0), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[39]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(cumsum(randn(100,3)*3, axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=1,q=0), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[40]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(cumsum(randn(100,3)*3, axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=1,q=1), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[41]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(randn(100,3)*3)\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=0,q=1), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[45]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(cumsum(randn(100,3)*3, axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=1,q=3), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[48]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(cumsum(randn(100,3)*3, axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=2,q=1), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[49]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(cumsum(cumsum(randn(100,3)*3, axis=0),axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=2,q=0), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[50]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nY = pd.DataFrame(cumsum(cumsum(randn(100,3)*3, axis=0),axis=0))\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=2,q=1), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[99]:

m = """
data {
      int<lower=0> K;
      int<lower=0> N;
        real y[N]; 
        }
parameters {
      real alpha;
      real beta[K];
      real<lower=0> sigma;
} model {
    alpha ~ normal(0,1);
    beta ~ normal(0,1);
    sigma ~ normal(0,1);
    
      for (n in (K+1):N) {
        real mu;
        mu = alpha;
        for (k in 1:K)
          mu = mu + beta[k] * y[n-k];
        y[n] ~ normal(mu, sigma);
} 
}
"""
model = StanModel(model_code=m)


# In[100]:

# Y = pd.DataFrame(cumsum(cumsum(randn(1000,3), axis=0),axis=0))
# Y = pd.DataFrame(randn(1000))
# Y.iloc[2:] += Y.iloc[:-2] + Y.iloc[1:-1]
# Y = pd.DataFrame(cumsum(randn(1000,3), axis=0))

n = 1000
Y = zeros(n)
Y[0] = randn()
Y[1] = randn()+.5*Y[0]
for i in range(2,n):
    Y[i] = randn()+Y[i-1]+.5*Y[i-2]
model_fit = model.sampling(data={'K': 2,
                    'N': n,
                    'y': Y}, n_jobs=n_jobs,iter=500)
print(model_fit)


# In[98]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\n# Y = pd.DataFrame(cumsum(cumsum(randn(100,3)*3, axis=0),axis=0))\nY = pd.DataFrame(Y)\nstan_data = {**models[model_name]['stan_data_creator'](Y,p=2,q=0), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[30]:

model_fit.plot(['mu', 'phi', 'theta'])


# In[20]:

model_fit.plot(['mu', 'phi'])


# In[ ]:

get_ipython().run_cell_magic('time', '', "\nmodel_name = 'Y~ARMA'\nstan_data = {**models[model_name]['stan_data_creator'](pd.DataFrame(rand(100,1)),p=0, q=0), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)\nprint(model_fit)")


# In[43]:

get_ipython().run_cell_magic('time', '', "Y = empirical_time_series.loc[1960:1970]\nany_data = Y.isnull().all(axis=0)\nY = Y[any_data[~any_data].index].iloc[:,[0,1,2,3,]]\n\nmodel_name = 'Y~ARMA'\nstan_data = {**models[model_name]['stan_data_creator'](Y,L=1), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)")


# In[ ]:

from scipy.stats import gaussian_kde
def predict_with_model(model_name, 
                       time_series,
                       predictors,
                       training_years,
                       horizons,
                       time_series_from_each_time_period,
                       technology_forecast_models_log_pd,
#                        technology_forecast_models_parameters,
                       technology_forecast_models_95CI,
#                        technology_forecast_models_Y_sim,
                       technology_forecast_models_fit, 
                       target_tech_names,
                       model_code=None, 
                       model_parameters=None,
                       parameter_priors=None,
                       print_output=True):
    
    if model_code is None:
        model_code = models[model_name]['code']
    if model_parameters is None:
        model_parameters = models[model_name]['model_parameters']
    if parameter_priors is None:
        parameter_priors = models[model_name]['parameter_priors']
    
    technology_forecast_models_log_pd[model_name] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
    technology_forecast_models_95CI[model_name] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
    
#     technology_forecast_models_parameters[model_name] = pd.Panel(items=target_tech_names,
#              major_axis=model_parameters, 
#              minor_axis=training_years)
#     technology_forecast_models_Y_sim[model_name] = {}
    technology_forecast_models_fit[model_name] = {}
    
    for training_year in training_years:
        print(training_year)

        forecast_start_ind = int(training_year-first_year)
        time_series_from_time_period = time_series_from_each_time_period[training_year]
        n_time_series_from_time_period = len(time_series_from_time_period)
        


        if predictors is not None:
            stan_data = stan_data_from_Y(time_series.loc[:training_year, 
                                                        time_series_from_time_period],
                                        forecast_to_observation=time_series.shape[0],
                                        predictors=predictors[time_series_from_time_period])
        else:
            stan_data = stan_data_from_Y(time_series.loc[:training_year, 
                                                    time_series_from_time_period],
                                    forecast_to_observation=time_series.shape[0])
        stan_data = {**stan_data, **parameter_priors} 

        ###
        model_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs)
        Y_sim = model_fit['Y_sim']
#         technology_forecast_models_Y_sim[model_name][training_year] = Y_sim
        
        if print_output:
            print(_print_stanfit(model_fit, model_parameters))
            
        technology_forecast_models_fit[model_name] = model_fit
#         for parameter in model_parameters:
#             technology_forecast_models_parameters[model_name]
#             p = model_fit[parameter].mean(axis=0)

#             if type(p)==numpy.ndarray:
#                 for i in range(len(p)):
#                     technology_forecast_models_parameters[model_name].ix[time_series_from_time_period, 
#                                                         parameter+'_%i'%i,
#                                                         training_year] = p[i]
#             else:        
#                 technology_forecast_models_parameters[model_name].ix[time_series_from_time_period, 
#                                                             parameter,
#                                                             training_year] = p

        for horizon in horizons:
            if horizon=='all':
                forecast_stop_ind = time_series.shape[0]
            else:
                forecast_stop_ind = horizon+forecast_start_ind
            
            times, techs = where(time_series[time_series_from_time_period].notnull())
            techs_to_forecast = techs[(forecast_start_ind<times)*(times<forecast_stop_ind)]
            times_to_forecast = times[(forecast_start_ind<times)*(times<forecast_stop_ind)]
            lpd = list(map(lambda x,y: x.logpdf(y)[0], 
                           map(gaussian_kde, Y_sim[:,times_to_forecast,techs_to_forecast].T), 
                           time_series[time_series_from_time_period].values[times_to_forecast, techs_to_forecast]))

            lpd = array(lpd)
            lpd[lpd==-inf] = log(finfo('d').tiny)
            lpd = pd.groupby(pd.Series(lpd),techs_to_forecast).sum()
            lpd = lpd.reindex(arange(len(time_series_from_time_period)))
            lpd.index = time_series_from_time_period
            technology_forecast_models_log_pd[model_name].ix[time_series_from_time_period,
                                                                 horizon,training_year] = lpd
            CI95 = portion_of_forecast_within_CI(model_fit, 'Y_sim', 
                                                 time_series[time_series_from_time_period].values, 
                                                 forecast_start_ind, 
                                                 forecast_stop_ind)
            technology_forecast_models_95CI[model_name].ix[time_series_from_time_period,
                                                           horizon,training_year] = CI95


# In[133]:

print(_print_stanfit(model_fit, pars=['mu', 'sigma']))

