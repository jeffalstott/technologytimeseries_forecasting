
# coding: utf-8

# In[2]:

from pystan import StanModel
n_jobs = 4
import pandas as pd
import seaborn as sns
sns.set_color_codes()
import pickle
get_ipython().magic('pylab inline')

models = pickle.load(open('model.pkl', 'rb'))


# In[18]:

def test_model_inference(model_name, Y=None, generated_data='data_latent', models=models, 
                         generator_iter=50, inference_iter=1000):
    
    if Y is None:
        Y = pd.DataFrame(rand(100,5))
    stan_data = {**models[model_name]['stan_data_creator'](Y, run_inference=False), 
                 **models[model_name]['parameter_priors']} 

    generated_example = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=generator_iter)

    sample = 20
    generated_parameters = {}
    for parameter in models[model_name]['model_parameters']:
        generated_parameters[parameter] = generated_example[parameter][sample]

    generated_data = pd.DataFrame(generated_example[generated_data][sample])

    stan_data = {**models[model_name]['stan_data_creator'](generated_data, run_inference=True), 
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


# In[19]:

model_name = 'Y_delta~N(mu,sigma) T[0,], missing data'
models[model_name] = {}

models[model_name]['code'] = """
functions {
  // lower bound is a, upper bound is b, rv is x, mean is mu, sd is sigma
  
  real alpha(real a, real mu, real sigma) {
    real out;
    out = (a==negative_infinity())? negative_infinity(): (a - mu)/sigma;
    return(out);
  }
  real beta(real b, real mu, real sigma) {
    real out;
    out = (b==positive_infinity())? positive_infinity(): (b - mu)/sigma;
    return(out);
  }
  real Z(real a, real b, real mu, real sigma) {
    return(normal_cdf(beta(b, mu, sigma), 0.0, 1.0) - normal_cdf(alpha(a, mu, sigma), 0.0, 1.0));
  }
  vector truncnorm_ng(vector p, real a, real b, real location, real scale) {
    vector[rows(p)] out;
    real tmp_Z;
    real tmp_alpha;
    
    tmp_alpha = normal_cdf(alpha(a, location, scale), 0, 1);
    tmp_Z = normal_cdf(beta(b, location, scale), 0, 1) - tmp_alpha;
    for(i in 1:rows(p)) {
      out[i] = inv_Phi(tmp_alpha + p[i]*tmp_Z)*scale + location;
    }
    return(out);
  }
}
data {
  int T; // number of rows
  int P; // number of columns
  matrix[T, P] Y; // -999 for missing values
  int run_inference;
  
  //priors
  real mu_location; 
  real mu_scale;
  real sigma_location;
  real sigma_scale;
    
}
parameters {
  matrix[T, P] z;
  vector[P] mu;
  vector<lower = 0>[P] sigma;
  corr_matrix[P] L_omega;
}
transformed parameters {
  matrix[T, P] theta;
  matrix[T, P] theta_constrained;
  matrix[T, P] data_latent;
  // use simple reparameterization to turn z into theta
  theta = z*cholesky_decompose(L_omega);
  for(p in 1:P){
    theta_constrained[1:T, p] = truncnorm_ng(Phi(col(theta, p)), 0, positive_infinity(), mu[p], sigma[p]);
  }
  
  //
  for(t in 1:T) {
    for(p in 1:P) {
      data_latent[t, p] = sum(theta_constrained[1:t, p]);
    }
  }
}
model {
  // priors
  to_vector(z) ~ normal(0, 1);
  mu ~ normal(mu_location, mu_scale);
  sigma ~ normal(sigma_location, sigma_scale);
  L_omega ~ lkj_corr(3);
  
  if(run_inference==1) {
    for(p in 1:P) {
    real tmp;
    tmp = 0.0;
    for(t in 1:T) {
      if(Y[t,p]>-998) {
        Y[t,p] ~ normal(data_latent[t, p], .1) T[tmp,];
        tmp = Y[t,p];
      }
     }
    }
  }
}
"""

models[model_name]['stan_model'] = StanModel(model_code=models[model_name]['code'])


# In[31]:

models[model_name]['parameter_priors'] = {
    'mu_location': 1,
    'mu_scale': .1,
    'sigma_location': 0,
    'sigma_scale': .1,
    }

models[model_name]['model_parameters'] = unique([i.rsplit('_', 1)[0] for i in models[model_name]['parameter_priors'].keys()])

def stan_data_creator(Y,run_inference=True):
    stan_data = {'Y':Y.fillna(-999),
                 'T': Y.shape[0],
                 'P': Y.shape[1],
                 'run_inference': int(run_inference),
                }
    return stan_data

models[model_name]['stan_data_creator'] = stan_data_creator

print(model_name)
print("Portion of parameters' true values within the 95%% CI: %.3f"%(test_model_inference(model_name)))


# In[35]:

Y = pd.DataFrame(rand(100,5))
stan_data = {**models[model_name]['stan_data_creator'](Y, run_inference=False), 
             **models[model_name]['parameter_priors']} 

generated_example = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=50)

sample = 20
generated_parameters = {}
for parameter in models[model_name]['model_parameters']:
    generated_parameters[parameter] = generated_example[parameter][sample]

generated_data = pd.DataFrame(generated_example['data_latent'][sample])


# In[37]:

generated_parameters


# In[42]:

stan_data = {**models[model_name]['stan_data_creator'](generated_data, run_inference=True), 
             **models[model_name]['parameter_priors']} 
model_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=1000)

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


# In[43]:

print(_print_stanfit(model_fit, pars=['mu', 'sigma']))


# In[ ]:

data_directory = '../data/'

empirical_data = pd.read_csv(data_directory+'time_series.csv',index_col=0)
empirical_data = empirical_data.reindex(arange(empirical_data.index[0],empirical_data.index[-1]+1))
metadata = pd.read_csv(data_directory+'time_series_metadata.csv')

target_tech_names = metadata.loc[(metadata['Type']=='Performance'), 'Name']
empirical_time_series = log(empirical_data[target_tech_names])

valid_time_series = sum(~empirical_time_series.loc[1976:].isnull())>3
valid_domains = metadata.set_index('Name').loc[valid_time_series.index[valid_time_series]]['Domain'].unique()

print("Number of valid domains: %i"%valid_domains.size)


# In[ ]:

get_ipython().run_cell_magic('time', '', "Y = empirical_time_series[valid_time_series].loc[1976:]\nany_data = Y.isnull().all(axis=0)\nY = Y[any_data[~any_data].index]\n\nmodel_name = 'Y_delta~N(mu,sigma) T[0,], missing data'\nstan_data = {**models[model_name]['stan_data_creator'](Y), **models[model_name]['parameter_priors']} \n\nmodel_fit = models[model_name]['stan_model'].sampling(data=stan_data, n_jobs=n_jobs,iter=500)")

