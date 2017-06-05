
# coding: utf-8

# Setup Packages and Functions
# ===

# In[386]:

# import pystan
import stanity
n_jobs = 4
import pandas as pd
import seaborn as sns
sns.set_color_codes()
get_ipython().magic('pylab inline')
from scipy.stats import norm, truncnorm, multivariate_normal, lognorm


# In[387]:

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
    
def plot_time_series_inference(model_fit, var='Y_sim', x=None,
                               ax=None, ind=0, **kwargs):
    from scipy.stats import scoreatpercentile
    ci_thresholds = [2.5, 25, 75, 97.5]
    if len(model_fit[var].shape)<3:
        data = model_fit[var]
    else:
        data = model_fit[var][:,ind,:]
    CIs = scoreatpercentile(data, ci_thresholds, axis=0)
    CIs = pd.DataFrame(data=CIs.T, columns=ci_thresholds)
    if ax is None:
        ax=gca()
    if x is None:
        x = arange(data.shape[1])
    ax.fill_between(x, CIs[2.5], CIs[97.5],alpha=.5, **kwargs)
    ax.fill_between(x, CIs[25], CIs[75], **kwargs)
    
from scipy.stats import percentileofscore
def portion_of_data_within_CI(model_fit, parameter, data, lower=2.5, upper=97.5):
    a = array((list(map(percentileofscore, model_fit[parameter].T, data))))
    return mean((lower<a)*(a<upper))

def calculate_Omega_from_L_Omega(model_fit):
    f = lambda x,y: matrix(x)*matrix(y)
    return list(map(f, model_fit['L_Omega'], transpose(model_fit['L_Omega'],[0,2,1])))


# Load Empirical Data and Look at It
# ===

# In[388]:

data_directory = '../data/'


# In[390]:

empirical_data = pd.read_csv(data_directory+'time_series.csv',index_col=0)
empirical_data = empirical_data.reindex(arange(empirical_data.index[0],empirical_data.index[-1]+1))
metadata = pd.read_csv(data_directory+'time_series_metadata.csv')

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']

figure()
q = empirical_data[target_tech_names]
r = q
r = r/r.apply(lambda x: x.ix[pd.Series.first_valid_index(x)])
r = log10(r)
r.plot(legend=False,kind='line')
xlabel("Year")
ylabel("log10(Performance),\nrelative to initial value")

figure()
z = log(q).apply(lambda x: (x.ix[x.last_valid_index()]-x.ix[x.first_valid_index()])/(x.last_valid_index()-x.first_valid_index()))
z.hist(bins=20)
xlabel("Mean Performance Increase (e^X)")
ylabel("p(Performance Increase)")

figure()
title("Example Technology:\n%s"%z.argmax())
n = (q)[z.argmax()].dropna()
n.plot(label='Empirical Data')
y = [n.iloc[0]]
for i in arange(n.index[0], n.index[-1]):
    y.append(y[-1]*exp(z.max()))
plot(arange(n.index[0], n.index[-1]+1), y, label='Constant Improvement Rate')
yscale('log')
legend(loc='upper left')
xlabel("Year")
ylabel("Performance")


# Define Single-Time-Series Models and Ensure They Correctly Fit to Simulated Data
# ===

# First, create some building blocks that we'll use in virtually all of our models
# ----

# In[4]:

functions_string = """    
int first_observation_ind(vector my_array){
    int t;
    t = 1;
    while(my_array[t] < -900){
      t = t+1;
    }
    return t;
}

int last_observation_ind(vector my_array, int length){
    int last_observation;
    last_observation = 0; 
    for(t in 1:length){
      if(my_array[t] > -900){
          last_observation = t;
      }
    }
    return last_observation;
}


int count_n_observations(vector my_array) {
    int count;
    count = 0;
    for (t in 1:num_elements(my_array)) {
        if(my_array[t] > -900){
            count = count + 1;
        }
    }
    return count;
}

real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
}

real lower_truncated_normal_lpdf_vector(vector x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
}
"""

data_string = """
int N_time_periods; // number of time periods
vector[N_time_periods] Y; // value each time period 
int N_time_periods_for_inference;
%(priors)s
"""

transformed_data_string="""
int first_observation;
int last_observation;
int N_observed_time_periods_for_inference;
int r_offset;
int n_observations;

first_observation = first_observation_ind(Y[1:N_time_periods_for_inference]);
last_observation = last_observation_ind(Y[1:N_time_periods_for_inference], 
                  N_time_periods_for_inference);

N_observed_time_periods_for_inference = last_observation-first_observation + 1;
r_offset = first_observation-1;

n_observations = count_n_observations(Y);
"""

parameters_string="""
vector<lower = 0,upper = 1>[N_observed_time_periods_for_inference-1] r_raw; // updates
%(parameters)s
"""

transformed_parameters_string="""
// Identify where the first and last non-missing data points are in Y
vector<lower = 0>[N_observed_time_periods_for_inference-1] r; // updates

{
// Dictate that the total change between each pair of observations is equal to the observed change between them
// This is relevant for time periods with missing data
int most_recent_observation;
most_recent_observation = first_observation;
for(t in first_observation+1:last_observation) {
  if(Y[t] > -900) {
    r[(most_recent_observation-r_offset):((t-1)-r_offset)] = 
    r_raw[(most_recent_observation-r_offset):((t-1)-r_offset)] /
    sum(r_raw[(most_recent_observation-r_offset):((t-1)-r_offset)]) * 
    (Y[t]-Y[most_recent_observation]);
    most_recent_observation = t;
    }
}
}
"""

generated_quantities_string = """
vector[N_time_periods] Y_sim;
vector[N_time_periods] log_likelihood;
real mean_change;
real variance_change;

mean_change = mean(r);
variance_change = variance(r);

//Fill out data in the missing periods
for(t in first_observation:last_observation) {
  if(Y[t] > -900){
      Y_sim[t] = Y[t];
  } else{
      Y_sim[t] = Y_sim[t-1] + r[(t-1)-r_offset];
  } 
}
{
real increase_size;
//Fill out future data points
for(t in last_observation+1:N_time_periods){
    // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
    // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
    increase_size = -1.0;  
    while (increase_size<0){
        increase_size = %(increase_size)s;
    }
    Y_sim[t] = increase_size + Y_sim[t-1];
}
}

//Fill out past data points
{
int t;
real increase_size;
t = first_observation;
while(t>1){
    increase_size = -1.0;  
    while (increase_size<0){
        increase_size = %(increase_size)s;
    }
    Y_sim[t-1] = Y_sim[t] - increase_size;
    t = t-1;
}
}

for(t in 2:N_time_periods){
    if(Y[t] > -900){
            %(log_likelihood)s;
    }
}
"""

basic_model_string = """
functions{
%(functions_string)s
}

data {
%(data_string)s
}

transformed data {
%(transformed_data_string)s
}
  
parameters {
%(parameters_string)s
}

transformed parameters {
%(transformed_parameters_string)s
}

model {
%(model_string)s
}

generated quantities {
%(generated_quantities_string)s
}
"""%{
'functions_string': functions_string,
'data_string': data_string,
'transformed_data_string': transformed_data_string,
'parameters_string': parameters_string,
'transformed_parameters_string': transformed_parameters_string,
'model_string': '%(model)s',
'generated_quantities_string': generated_quantities_string}


# Next, define our models, using those building blocks
# ---

# In[5]:

model_code = {}

model_code['improvement~N(mu,sigma)'] = basic_model_string%{
    'model': """
            mu ~ normal(mu_prior_location, mu_prior_scale);
            sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);

            for(t in 1:N_observed_time_periods_for_inference-1){
                target += lower_truncated_normal_lpdf(r[t] | mu, sigma, 0);
            }""",
    'parameters': """
                    real<lower = 0> mu;
                    real<lower = 0> sigma;
                    """,
    'increase_size': 'normal_rng(mu,sigma)',
    'log_likelihood': 'log_likelihood[t] = lower_truncated_normal_lpdf(Y[t]-Y_sim[t-1]| mu, sigma, 0)',
    'priors': """
            real mu_prior_location;
            real mu_prior_scale;  
            real sigma_prior_location;
            real sigma_prior_scale;"""}

model_code['improvement~p(theta)N(mu,sigma)'] = basic_model_string%{
    'model': """
            mu ~ normal(mu_prior_location, mu_prior_scale);
            sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
            theta ~ normal(theta_prior_location, theta_prior_scale);

            for(t in 1:N_observed_time_periods_for_inference-1){
                target += log_mix(theta, lower_truncated_normal_lpdf(r[t] | mu, sigma, 0), 
                              lower_truncated_normal_lpdf(r[t] | 0, 0.01, 0));
            }""",
    'parameters': """
                real<lower = 0> mu;
                real<lower = 0> sigma;
                real<lower = 0, upper = 1> theta;
                """,
    'increase_size': 'bernoulli_rng(theta) * normal_rng(mu,sigma)',
    'log_likelihood': """
                if((Y[t]-Y_sim[t-1])==0) {
                log_likelihood[t] = log(1-theta);
                } else {
                    log_likelihood[t] = log(theta) + lower_truncated_normal_lpdf(Y[t]-Y_sim[t-1]| mu, sigma, 0);
                }
                """,
    'priors': """
            real mu_prior_location;
            real mu_prior_scale;

            real sigma_prior_location;
            real sigma_prior_scale;

            real theta_prior_location;
            real theta_prior_scale;"""}


# Test the models on simulated data
# ---

# In[6]:

### Random walk, missing data, positive steps
p_missing = 0.3
mu = 5
sigma = 1
n = 100


a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
missing[-1] = False
time_series[missing]=nan


hist(data, bins=50)
figure()
plot(time_series)

n_past_steps = 25
n_future_steps = 25
stan_data = {'N_time_periods': len(time_series)+n_future_steps+n_past_steps,
            'N_time_periods_for_inference': len(time_series)+n_past_steps,
             'Y': pd.Series(concatenate((empty(n_past_steps)*nan,
                                         time_series,
                                         empty(n_future_steps)*nan),0)).fillna(-999),
            'mu_prior_location': 0,
            'mu_prior_scale': 1,
            'sigma_prior_location': 0,
            'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code['improvement~N(mu,sigma)'], data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma']))
print("mu: %.2f, inferred mu: %.2f"%(mu, model_fit['mu'].mean()))
print("sigma: %.2f, inferred sigma: %.2f"%(sigma, model_fit['sigma'].mean()))

figure()
plot_time_series_inference(model_fit)
scatter(arange(len(time_series))+n_past_steps, time_series,s=2)
xlim(xmin=0)
# ylim(ymin=0)


# In[7]:

### Random walk, missing data, positive steps, possibility of no step, small noise on no step
p_missing = 0.3
theta = .8
mu = 5
sigma = 1
n = 100

improvement = rand(n)<theta
improvement[0] = True

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)
data[~improvement]=0

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
missing[-1] = False
time_series[missing]=nan


hist(data, bins=50)
figure()
plot(time_series)



n_past_steps = 25
n_future_steps = 25
stan_data = {'N_time_periods': len(time_series)+n_future_steps+n_past_steps,
            'N_time_periods_for_inference': len(time_series)+n_past_steps,
             'Y': pd.Series(concatenate((empty(n_past_steps)*nan,
                                         time_series,
                                         empty(n_future_steps)*nan),0)).fillna(-999),
           'theta_prior_location': .8,
           'theta_prior_scale': 2,
            'mu_prior_location': 3,
            'mu_prior_scale': 1,
            'sigma_prior_location': 1,
            'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code['improvement~p(theta)N(mu,sigma)'], data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma','theta']))

print("mu: %.2f, inferred mu: %.2f"%(mu, model_fit['mu'].mean()))
print("sigma: %.2f, inferred sigma: %.2f"%(sigma, model_fit['sigma'].mean()))
print("theta: %.2f, inferred theta: %.2f"%(theta, model_fit['theta'].mean()))

figure()
plot_time_series_inference(model_fit)
scatter(arange(len(time_series))+n_past_steps, time_series,s=2)
xlim(xmin=0)
# ylim(ymin=0)


# Define Multiple Time-Series Models (pooled and unpooled) and Ensure They Correctly Fit to Simulated Data
# ===

# Create building blocks
# ---

# In[8]:

multiple_data_string = """
int N_technologies;
int N_time_periods; // number of time periods
vector[N_time_periods] Y[N_technologies]; // value each time period

int N_time_periods_for_inference;
%(priors)s
"""

multiple_transformed_data_string="""
int first_observation[N_technologies];
int last_observation[N_technologies];
int N_observed_time_periods_for_inference[N_technologies];
int r_observation_offset[N_technologies];
int n_observations[N_technologies];
int r_array_offset[N_technologies];

for (tech in 1:N_technologies){
  first_observation[tech] = first_observation_ind(Y[tech][1:N_time_periods_for_inference]);
  last_observation[tech] = last_observation_ind(Y[tech][1:N_time_periods_for_inference], 
                      N_time_periods_for_inference);

  N_observed_time_periods_for_inference[tech] = last_observation[tech]-first_observation[tech] + 1;
  r_observation_offset[tech] = first_observation[tech]-1;
  n_observations[tech] = count_n_observations(Y[tech]);
}
r_array_offset[1] = 0;
for (tech in 2:N_technologies){
    r_array_offset[tech] = N_observed_time_periods_for_inference[tech-1]+r_array_offset[tech-1]-1;
}
"""

multiple_parameters_string="""
vector<lower = 0,upper = 1>[sum(N_observed_time_periods_for_inference)-N_technologies] r_raw; // updates
%(parameters)s
"""

multiple_transformed_parameters_string="""
// Identify where the first and last non-missing data points are in Y
vector<lower = 0>[sum(N_observed_time_periods_for_inference)-
                            N_technologies] r; // updates

{
// Dictate that the total change between each pair of observations is equal to the observed change between them
// This is relevant for time periods with missing data
int most_recent_observation;
for (tech in 1:N_technologies){
  most_recent_observation = first_observation[tech];
  for(t in first_observation[tech]+1:last_observation[tech]) {
      if(Y[tech][t] > -900) {
        r[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
          (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] = 
        r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
          (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] /
        sum(r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
          (r_array_offset[tech]+(t-1)-r_observation_offset[tech])]) * 
        (Y[tech][t]-Y[tech][most_recent_observation]);
        most_recent_observation = t;
        }
    }
  }
}
"""

multiple_generated_quantities_string = """
vector[N_time_periods] Y_sim[N_technologies];
vector[N_time_periods] log_likelihood[N_technologies];
//real mean_change[N_technologies];
//real variance_change[N_technologies];

//for (tech in 1:N_technologies){
//    mean_change = mean(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
//    variance_change = variance(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
//}

//Fill out data in the missing periods
for (tech in 1:N_technologies){
    for(t in first_observation[tech]:last_observation[tech]) {
      if(Y[tech][t] > -900){
          Y_sim[tech][t] = Y[tech][t];
      } else{
          Y_sim[tech][t] = Y_sim[tech][t-1] + r[r_array_offset[tech]+(t-1)-r_observation_offset[tech]];
      } 
    }
}
{
real increase_size;
//Fill out future data points
for (tech in 1:N_technologies){
    for(t in last_observation[tech]+1:N_time_periods){
        // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
        // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
        increase_size = -1.0;  
        while (increase_size<0){
            increase_size = %(increase_size)s;
        }
        Y_sim[tech][t] = increase_size + Y_sim[tech][t-1];
    }
}
}

//Fill out past data points
{
int t;
real increase_size;
for (tech in 1:N_technologies){
    t = first_observation[tech];
    while(t>1){
        increase_size = -1.0;  
        while (increase_size<0){
            increase_size = %(increase_size)s;
        }
        Y_sim[tech][t-1] = Y_sim[tech][t] - increase_size;
        t = t-1;
    }
}
}

for (tech in 1:N_technologies){
    for(t in 2:N_time_periods){
        if(Y[tech][t] > -900){
                %(log_likelihood)s
        }
    }
}
"""

multiple_model_string = """
functions{
%(functions_string)s
}

data {
%(data_string)s
}

transformed data {
%(transformed_data_string)s
}
  
parameters {
%(parameters_string)s
}

transformed parameters {
%(transformed_parameters_string)s
}

model {
%(model_string)s
}

generated quantities {
%(generated_quantities_string)s
}
"""%{
'functions_string': functions_string,
'data_string': multiple_data_string,
'transformed_data_string': multiple_transformed_data_string,
'parameters_string': multiple_parameters_string,
'transformed_parameters_string': multiple_transformed_parameters_string,
'model_string': '%(model)s',
'generated_quantities_string': multiple_generated_quantities_string}


# Define models
# ---

# In[383]:

###### improvement~N(mu,sigma), multiple (unpooled)

model_code['improvement~N(mu,sigma), multiple'] = multiple_model_string%{
    'model': """
            mu ~ normal(mu_prior_location, mu_prior_scale);
            sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);

            for (tech in 1:N_technologies){
                for(t in 1:N_observed_time_periods_for_inference[tech]-1){
                    target += lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0);
                }
            }""",
    'parameters': """
                real<lower = 0> mu[N_technologies];
                real<lower = 0> sigma[N_technologies];
                """,
    'increase_size': 'normal_rng(mu[tech],sigma[tech])',
    'log_likelihood': """
                    log_likelihood[tech][t] = lower_truncated_normal_lpdf(Y[tech][t]-
                                                                    Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);;
                """,
    'priors': """
            real mu_prior_location;
            real mu_prior_scale;

            real sigma_prior_location;
            real sigma_prior_scale;
            """}




###### improvement~N(mu,sigma), hierarchical
model_code['improvement~N(mu,sigma), hierarchical'] = multiple_model_string%{
    'model': """
            tau ~ cauchy(0, 2);
            //Omega ~ lkj_corr(1);
            L_Omega ~ lkj_corr_cholesky(1);
            //z ~ normal(0,1);
            mu_mu ~ normal(mu_prior_location, mu_prior_scale);
            mu_sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
            
            {
            matrix[N_technologies, 2] parvec;
            vector[2] mu_parvec;

            parvec = append_col(mu, sigma);
            mu_parvec[1] = mu_mu;
            mu_parvec[2] = mu_sigma;
            
            for (tech in 1:N_technologies){
                //parvec[tech] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
                //parvec[tech] = (mu_parvec + (diag_pre_multiply(tau,L_Omega) * z))';
                log(parvec[tech]) ~ multi_normal_cholesky(log(mu_parvec), diag_pre_multiply(tau, L_Omega));
                for(t in 1:N_observed_time_periods_for_inference[tech]-1){
                    target += lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0);
                }
                //target += lower_truncated_normal_lpdf_vector(r[(r_array_offset[tech]+1):(r_array_offset[tech]+N_observed_time_periods_for_inference[tech]-1)] | mu[tech], sigma[tech], 0);
            }
            }""",
    'parameters': """
                vector<lower = 0>[N_technologies] mu;
                vector<lower = 0>[N_technologies] sigma;
                
                //corr_matrix[2] Omega;
                cholesky_factor_corr[2] L_Omega;
                vector<lower = 0>[2] tau;
                //vector<lower = 0>[2] z;
                
                real<lower = 0> mu_mu;
                real<lower = 0> mu_sigma;
                """,
    'increase_size': 'normal_rng(mu[tech],sigma[tech])',
    'log_likelihood': """
                    log_likelihood[tech][t] = lower_truncated_normal_lpdf(Y[tech][t]-
                                                                    Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);
                """,
    'priors': """
            real mu_prior_location;
            real mu_prior_scale;

            real sigma_prior_location;
            real sigma_prior_scale;
            """}


###### improvement~logN(mu,sigma), hierarchical
model_code['improvement~logN(mu,sigma), hierarchical'] = multiple_model_string%{
    'model': """
            tau ~ cauchy(0, 2);
            //Omega ~ lkj_corr(1);
            L_Omega ~ lkj_corr_cholesky(1);
            //z ~ normal(0,1);
            mu_mu ~ normal(mu_prior_location, mu_prior_scale);
            mu_sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
            
            {
            matrix[N_technologies, 2] parvec;
            vector[2] mu_parvec;

            parvec = append_col(mu, sigma);
            mu_parvec[1] = mu_mu;
            mu_parvec[2] = mu_sigma;
            
            for (tech in 1:N_technologies){
                //parvec[tech] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
                //parvec[tech] = (mu_parvec + (diag_pre_multiply(tau,L_Omega) * z))';
                parvec[tech] ~ multi_normal_cholesky(mu_parvec, diag_pre_multiply(tau, L_Omega));
                for(t in 1:N_observed_time_periods_for_inference[tech]-1){
                    target += lognormal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech]);
                }
            }
            }""",
    'parameters': """
                vector[N_technologies] mu;
                vector<lower = 0>[N_technologies] sigma;
                
                //corr_matrix[2] Omega;
                cholesky_factor_corr[2] L_Omega;
                vector<lower = 0>[2] tau;
                //vector<lower = 0>[2] z;
                
                real mu_mu;
                real<lower = 0> mu_sigma;
                """,
    'increase_size': 'lognormal_rng(mu[tech],sigma[tech])',
    'log_likelihood': """
                    log_likelihood[tech][t] = lognormal_lpdf(Y[tech][t]-
                                                                    Y_sim[tech][t-1]| mu[tech], sigma[tech]);
                """,
    'priors': """
            real mu_prior_location;
            real mu_prior_scale;

            real sigma_prior_location;
            real sigma_prior_scale;
            """}


###### improvement~p(theta)N(mu,sigma), hierarchical
model_code['improvement~p(theta)N(mu,sigma), hierarchical'] = multiple_model_string%{
    'model': """
            tau ~ cauchy(0, 2);
            Omega ~ lkj_corr(1);
            mu_mu ~ normal(mu_prior_location, mu_prior_scale);
            mu_sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
            mu_theta ~ normal(mu_prior_location, mu_prior_scale);

            {
            matrix[N_technologies, 3] parvec;
            vector[3] mu_parvec;

            parvec = append_col(mu, sigma);
            mu_parvec[1] = mu_mu;
            mu_parvec[2] = mu_sigma;
            mu_parvec[3] = mu_theta;
            
            for (tech in 1:N_technologies){
                parvec[tech] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
                for(t in 1:N_observed_time_periods_for_inference[tech]-1){
                    target += log_mix(theta[tech], lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0), 
                              lower_truncated_normal_lpdfr[r_array_offset[tech]+t] | 0, 0.01, 0));
                }
            }
            }""",
    'parameters': """
                vector<lower = 0>[N_technologies] mu;
                vector<lower = 0>[N_technologies] sigma;
                real<lower = 0, upper = 1> theta[N_technologies];
                
                corr_matrix[3] Omega;
                vector<lower = 0>[3] tau;
                
                real<lower = 0> mu_mu;
                real<lower = 0> mu_sigma;
                real<lower = 0, upper = 1> mu_theta;
                """,
    'increase_size': 'bernoulli_rng(theta[tech]) * normal_rng(mu[tech],sigma[tech])',
    'log_likelihood': """
                    if((Y[tech][t]-Y_sim[tech][t-1])==0) {
                        log_likelihood[tech][t] = log(1-theta[tech]);
                    } else {
                        log_likelihood[tech][t] = log(theta[tech]) + 
                        lower_truncated_normal_lpdf(Y[tech][t]-Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);
                    }
                """,
    'priors': """
            real mu_prior_location;
            real mu_prior_scale;

            real sigma_prior_location;
            real sigma_prior_scale;
            
            real theta_prior_location;
            real theta_prior_scale;
            """}


# Test on simulated data
# ---

# In[25]:

### Random walk, missing data, positive steps, multiple technologies
p_missing = 0.3
mu = 5
sigma = 1
n = 100


a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
missing[-1] = False
time_series[missing]=nan

time_series0 = time_series


p_missing = 0.2
mu = 3
sigma = 2
n = 100


a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
missing[-1] = False
time_series[missing]=nan

time_series1 = time_series


stan_data = {'N_technologies': 2,
            'N_time_periods': n,
            'N_time_periods_for_inference': n,
             'Y': pd.DataFrame([time_series0, time_series1]).fillna(-999).values,
            'mu_prior_location': 3,
            'mu_prior_scale': 1,
            'sigma_prior_location': 1,
            'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code['improvement~N(mu,sigma), multiple'], data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma']))


# In[107]:

### Random walk, missing data, positive steps, multiple technologies, hierarchical
mu_mu = .3
mu_sigma = .5

Omega = matrix([[1,.7],
               [.7,1]])
tau = array([1,1])
cov = diag(tau)*Omega*diag(tau)

p_missing = 0
n = 100
N_technologies = 50

mus = zeros(N_technologies)
sigmas = zeros(N_technologies)

time_series = empty((n, N_technologies))
for i in arange(N_technologies):
    while mus[i]==0:
        mu, sigma = multivariate_normal(array([mu_mu, mu_sigma]), cov=cov).rvs(1)
        if mu>0 and sigma>0:
            mus[i] = mu
            sigmas[i] = sigma
    a = -mu / sigma
    data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)
    time_series[:,i] = cumsum(data)
    missing = rand(n)<p_missing
    time_series[missing,i]=nan


stan_data = {'N_technologies': N_technologies,
            'N_time_periods': n,
            'N_time_periods_for_inference': n,
             'Y': pd.DataFrame(time_series).T.fillna(-999).values,
            'mu_prior_location': .1,
            'mu_prior_scale': 1,
            'sigma_prior_location': 0,
            'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code['improvement~N(mu,sigma), hierarchical'], data=stan_data, n_jobs=n_jobs)

print(portion_of_data_within_CI(model_fit, 'mu', mus, 2.5, 97.5))
print(portion_of_data_within_CI(model_fit, 'sigma', sigmas, 2.5, 97.5))

print(mean(calculate_Omega_from_L_Omega(model_fit),axis=0))

print(_print_stanfit(model_fit, ['mu', 'sigma', 'L_Omega', 'tau']))

scatter(model_fit['mu'].mean(axis=0),
        model_fit['sigma'].mean(axis=0)
       )


# In[384]:

### Random walk, missing data, lognormal steps, multiple technologies, hierarchical
mu_mu = .3
mu_sigma = .5

Omega = matrix([[1,.7],
               [.7,1]])
tau = array([1,1])
cov = diag(tau)*Omega*diag(tau)

p_missing = 0
n = 100
N_technologies = 50

mus = zeros(N_technologies)
sigmas = zeros(N_technologies)

time_series = empty((n, N_technologies))
for i in arange(N_technologies):
    while mus[i]==0:
        mu, sigma = multivariate_normal(array([mu_mu, mu_sigma]), cov=cov).rvs(1)
        if sigma>0:
            mus[i] = mu
            sigmas[i] = sigma
    data = lognorm(sigma, scale=exp(mu), loc=0).rvs(n)
    time_series[:,i] = cumsum(data)
    missing = rand(n)<p_missing
    time_series[missing,i]=nan


stan_data = {'N_technologies': N_technologies,
            'N_time_periods': n,
            'N_time_periods_for_inference': n,
             'Y': pd.DataFrame(time_series).T.fillna(-999).values,
            'mu_prior_location': .1,
            'mu_prior_scale': 1,
            'sigma_prior_location': 0,
            'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code['improvement~logN(mu,sigma), hierarchical'], data=stan_data, n_jobs=n_jobs)


print(portion_of_data_within_CI(model_fit, 'mu', mus, 2.5, 97.5))
print(portion_of_data_within_CI(model_fit, 'sigma', sigmas, 2.5, 97.5))

print(mean(calculate_Omega_from_L_Omega(model_fit),axis=0))

print(_print_stanfit(model_fit, ['mu', 'sigma', 'L_Omega', 'tau']))


# In[385]:

scatter(model_fit['mu'].mean(axis=0),
        model_fit['sigma'].mean(axis=0)
       )


# Run Time-Series Models on Empirical Performance Data
# ====

# In[392]:

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']
print("%i technologies"%target_tech_names.shape[0])

time_series = log(empirical_data[target_tech_names])


# In[393]:

technology_models_likelihood = {}
technology_models_parameters = {}


# In[ ]:

model_type = 'improvement~N(mu,sigma), multiple'
model_parameters = ['mu', 'sigma']
parameter_priors = {'mu_prior_location': 0,
                    'mu_prior_scale': 1,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                }
m = lambda df: truncnorm(0, 10000, loc=df['mu'], scale=df['sigma']).mean()
v = lambda df: truncnorm(0, 10000, loc=df['mu'], scale=df['sigma']).var()

technology_models_log_likelihood[model_type] = pd.Series(index=target_tech_names)

technology_models_parameters[model_type] = pd.DataFrame(index=target_tech_names,
         columns=model_parameters)


stan_data = {'N_technologies': time_series.shape[1],
             'N_time_periods': time_series.shape[0],
        'N_time_periods_for_inference': time_series.shape[0],
         'Y': time_series.fillna(-999).T,
                 }
stan_data = {**stan_data, **parameter_priors} 

###
model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
print(_print_stanfit(model_fit, model_parameters))

technology_models_log_likelihood[model_type] = nanmean(nanmean(model_fit['log_likelihood'],axis=0),axis=1)

for parameter in model_parameters:
    technology_models_parameters[model_type][parameter] = model_fit[parameter].mean(axis=0)
    

technology_models_parameters[model_type]['mean']=technology_models_parameters[model_type].apply(m,axis=1)
technology_models_parameters[model_type]['variance']=technology_models_parameters[model_type].apply(v,axis=1)


# In[245]:

model_type = 'improvement~N(mu,sigma), hierarchical'
model_parameters = ['mu', 'sigma']
parameter_priors = {'mu_prior_location': .3,
                    'mu_prior_scale': 3,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                }
m = lambda df: truncnorm(0, 10000, loc=df['mu'], scale=df['sigma']).mean()
v = lambda df: truncnorm(0, 10000, loc=df['mu'], scale=df['sigma']).var()

technology_models_log_likelihood[model_type] = pd.Series(index=target_tech_names)
technology_models_parameters[model_type] = pd.DataFrame(index=target_tech_names,
         columns=model_parameters)


stan_data = {'N_technologies': time_series.shape[1],
             'N_time_periods': time_series.shape[0],
        'N_time_periods_for_inference': time_series.shape[0],
         'Y': time_series.fillna(-999).T,
                 }
stan_data = {**stan_data, **parameter_priors} 

###
model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
print(_print_stanfit(model_fit, model_parameters))

technology_models_log_likelihood[model_type] = nanmean(nanmean(model_fit['log_likelihood'],axis=0),axis=1)

for parameter in model_parameters:
    technology_models_parameters[model_type][parameter] = model_fit[parameter].mean(axis=0)
    

technology_models_parameters[model_type]['mean']=technology_models_parameters[model_type].apply(m,axis=1)
technology_models_parameters[model_type]['variance']=technology_models_parameters[model_type].apply(v,axis=1)


# In[246]:

f = lambda x,y: matrix(x)*matrix(y)
r = list(map(f, model_fit['L_Omega'], transpose(model_fit['L_Omega'],[0,2,1])))
mean(r,axis=0)


# In[351]:

model_type = 'improvement~logN(mu,sigma), hierarchical'
model_parameters = ['mu', 'sigma']
parameter_priors = {'mu_prior_location': 0,
                    'mu_prior_scale': 1,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                }
m = lambda df: lognorm(df['sigma'], scale=exp(df['mu']), loc=0).mean()
v = lambda df: lognorm(df['sigma'], scale=exp(df['mu']), loc=0).var()

technology_models_log_likelihood[model_type] = pd.Series(index=target_tech_names)
technology_models_parameters[model_type] = pd.DataFrame(index=target_tech_names,
         columns=model_parameters)


stan_data = {'N_technologies': time_series.shape[1],
             'N_time_periods': time_series.shape[0],
        'N_time_periods_for_inference': time_series.shape[0],
         'Y': time_series.fillna(-999).T,
                 }
stan_data = {**stan_data, **parameter_priors} 

###
model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
print(_print_stanfit(model_fit, model_parameters))

technology_models_log_likelihood[model_type] = nanmean(nanmean(model_fit['log_likelihood'],axis=0),axis=1)

for parameter in model_parameters:
    technology_models_parameters[model_type][parameter] = model_fit[parameter].mean(axis=0)
    

technology_models_parameters[model_type]['mean']=technology_models_parameters[model_type].apply(m,axis=1)
technology_models_parameters[model_type]['variance']=technology_models_parameters[model_type].apply(v,axis=1)


# In[352]:

f = lambda x,y: matrix(x)*matrix(y)
r = list(map(f, model_fit['L_Omega'], transpose(model_fit['L_Omega'],[0,2,1])))
mean(r,axis=0)


# In[249]:

pd.DataFrame(technology_models_log_likelihood)


# In[250]:

pd.DataFrame(technology_models_log_likelihood).plot('improvement~N(mu,sigma), multiple',
                                                   'improvement~logN(mu,sigma), hierarchical', kind='scatter')
plot(xlim(), xlim())


# In[251]:

for model_type in technology_models_parameters.keys():
    technology_models_parameters[model_type].plot('mu', 'sigma',kind='scatter')
    title(model_type)
    technology_models_parameters[model_type].plot('mean', 'variance',kind='scatter')
    title(model_type)


# Use Models to Predict the Future
# ===

# In[364]:

training_years = arange(1950,2000,10)
horizons = [5,10,'all']


first_year = empirical_data.index[0]
time_series_from_each_time_period = {}

for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    ### Select only those time series that have at least 3 data points from before the training year
    time_series_from_time_period = time_series.columns[time_series.iloc[:start_ind].notnull().sum(axis=0)>=3]
    time_series_from_each_time_period[training_year] = time_series_from_time_period


# In[253]:

model_type = 'improvement~N(mu,sigma), multiple'
model_parameters = ['mu', 'sigma']
parameter_priors = {'mu_prior_location': .1,
                    'mu_prior_scale': 3,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                }

technology_models_prediction[model_type] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
technology_models_parameters[model_type] = pd.Panel(items=target_tech_names,
         major_axis=model_parameters, 
         minor_axis=training_years)


for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    time_series_from_time_period = time_series_from_each_time_period[training_year]
    n_time_series_from_time_period = len(time_series_from_time_period)
    
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                     }
    stan_data = {**stan_data, **parameter_priors} 

    ###
    model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, model_parameters))
    for parameter in model_parameters:
        technology_models_parameters[model_type].ix[time_series_from_time_period, 
                                                    parameter,
                                                    training_year] = model_fit[parameter].mean(axis=0)
    
    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll


# In[254]:

model_type = 'improvement~N(mu,sigma), hierarchical'
model_parameters = ['mu', 'sigma']
parameter_priors = {'mu_prior_location': 0,
                    'mu_prior_scale': 3,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                }

technology_models_prediction[model_type] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
technology_models_parameters[model_type] = pd.Panel(items=target_tech_names,
         major_axis=model_parameters, 
         minor_axis=training_years)


for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    ### Select only those time series that have at least 3 data points from before the training year
    time_series_from_time_period = time_series_from_each_time_period[training_year]
    n_time_series_from_time_period = len(time_series_from_time_period)
    
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                     }
    stan_data = {**stan_data, **parameter_priors} 

    ###
    model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, model_parameters))
    for parameter in model_parameters:
        technology_models_parameters[model_type].ix[time_series_from_time_period, 
                                                    parameter,
                                                    training_year] = model_fit[parameter].mean(axis=0)
    
    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll


# In[369]:

model_type = 'improvement~logN(mu,sigma), hierarchical'
model_parameters = ['mu', 'sigma']
parameter_priors = {'mu_prior_location': 0,
                    'mu_prior_scale': 1,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                }

technology_models_prediction[model_type] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
technology_models_parameters[model_type] = pd.Panel(items=target_tech_names,
         major_axis=model_parameters, 
         minor_axis=training_years)


for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    ### Select only those time series that have at least 3 data points from before the training year
    time_series_from_time_period = time_series_from_each_time_period[training_year]
    n_time_series_from_time_period = len(time_series_from_time_period)
    
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                     }
    stan_data = {**stan_data, **parameter_priors} 

    ###
    model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, model_parameters))
    for parameter in model_parameters:
        technology_models_parameters[model_type].ix[time_series_from_time_period, 
                                                    parameter,
                                                    training_year] = model_fit[parameter].mean(axis=0)
    
    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll


# In[279]:

q = pd.Panel4D(technology_models_prediction)
for i in arange(q.shape[2]):
    q.mean(axis=1).iloc[:,i].plot()
    title(str(q.major_axis[i])+' years forward')
    ylabel('mean log(likelihoood) of future observations')
    xlabel('Model trained up until this year')


# In[259]:

technology_models_prediction['improvement~N(mu,sigma), multiple'].mean(axis=0).T.plot()
yscale('symlog')

figure()

technology_models_prediction['improvement~N(mu,sigma), hierarchical'].mean(axis=0).T.plot()
yscale('symlog')


# In[256]:

technology_models_parameters['improvement~N(mu,sigma), multiple'].plot('mu', 'sigma', kind='scatter')
figure()
technology_models_parameters['improvement~N(mu,sigma), hierarchical'].plot('mu', 'sigma', kind='scatter')


# In[10]:

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']
print("%i technologies"%target_tech_names.shape[0])

technology_models['improvement~p(theta)N(mu,sigma)'] = pd.DataFrame(columns=['mu', 'sigma', 
                                                                             'theta', 
                                                                             'mean', 'variance',
                                                                             'mean_analytic', 'variance_analytic',
                                                                            'log_likelihood'], 
                                                                    index=target_tech_names)
technology_models['improvement~N(mu,sigma)'] = pd.DataFrame(columns=['mu', 'sigma', 
                                                                     'mean', 'variance',
                                                                    'mean_analytic', 'variance_analytic',
                                                                    'log_likelihood'], 
                                                            index=target_tech_names)

n_future_steps = 0
for tech_name in target_tech_names:
    print('------------------------------------')
    print(tech_name)
    print('------------------------------------')
    figure()
    title(tech_name)
    
    time_series = log10(data[tech_name]**-1)
    scatter(arange(len(time_series)), time_series,s=2)

    stan_data = {'N_time_periods': len(time_series),
            'N_time_periods_for_inference': len(time_series),
             'Y': pd.Series(concatenate((time_series,
                                         empty(n_future_steps)*nan),0)).fillna(-999),
            'mu_prior_location': .1,
            'mu_prior_scale': 2,
            'sigma_prior_location': 1,
            'sigma_prior_scale': 2}
    
    ###
    model_fit = stanity.fit(model_code['improvement~N(mu,sigma)'], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, ['mu', 'sigma']))
    technology_models['improvement~N(mu,sigma)'].ix[tech_name,'mu'] = model_fit['mu'].mean()
    technology_models['improvement~N(mu,sigma)'].ix[tech_name,'sigma'] = model_fit['sigma'].mean()
    technology_models['improvement~N(mu,sigma)'].ix[tech_name,'mean'] = model_fit['mean_change'].mean()
    technology_models['improvement~N(mu,sigma)'].ix[tech_name,'variance'] = model_fit['variance_change'].mean()
    technology_models['improvement~N(mu,sigma)'].ix[tech_name, 'mean_analytic'] = truncnorm(0, 10000, 
                                                         loc=model_fit['mu'].mean(), 
                                                         scale=model_fit['sigma'].mean()).mean()
    technology_models['improvement~N(mu,sigma)'].ix[tech_name, 'variance_analytic'] = truncnorm(0, 10000, 
                                                         loc=model_fit['mu'].mean(), 
                                                         scale=model_fit['sigma'].mean()).var()
    technology_models['improvement~N(mu,sigma)'].ix[tech_name,'log_likelihood'] = nanmean(model_fit['log_likelihood'])

    plot_time_series_inference(model_fit, color='g')
    
    ###
    stan_data['theta_prior_location'] = .5
    stan_data['theta_prior_scale'] = 2
    
    model_fit = stanity.fit(model_code['improvement~p(theta)N(mu,sigma)'], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, ['mu', 'sigma','theta']))
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name,'mu'] = model_fit['mu'].mean()
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name,'sigma'] = model_fit['sigma'].mean()
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name,'theta'] = model_fit['theta'].mean()
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name,'mean'] = model_fit['mean_change'].mean()
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name,'variance'] = model_fit['variance_change'].mean()
    
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name, 'mean_analytic'] = model_fit['theta'].mean() * truncnorm(0, 10000, 
                                                         loc=model_fit['mu'].mean(), 
                                                         scale=model_fit['sigma'].mean()).mean()
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name, 'variance_analytic'] = model_fit['theta'].mean() * truncnorm(0, 10000, 
                                                         loc=model_fit['mu'].mean(), 
                                                         scale=model_fit['sigma'].mean()).var()
    technology_models['improvement~p(theta)N(mu,sigma)'].ix[tech_name,'log_likelihood'] = nanmean(model_fit['log_likelihood'])
        
    plot_time_series_inference(model_fit, color='b')


# In[11]:

technology_models['improvement~N(mu,sigma)'].plot('mean', 'variance', kind='scatter')
figure()
technology_models['improvement~p(theta)N(mu,sigma)'].plot('mean', 'variance', kind='scatter')


# In[12]:

technology_models['improvement~N(mu,sigma)'].plot('mean_analytic', 'variance_analytic', kind='scatter')


# In[13]:

technology_models['improvement~N(mu,sigma)'].plot('mean_analytic', 'mean', kind='scatter')
figure()
technology_models['improvement~N(mu,sigma)'].plot('variance_analytic', 'variance', kind='scatter')


# Fit all time series using data up until specific years, then predict future years
# ====

# In[14]:

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']
print("%i technologies"%target_tech_names.shape[0])

training_years = arange(1950,2000,10)
horizons = [5,10,'all']

technology_models_prediction['improvement~p(theta)N(mu,sigma)'] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
technology_models_prediction['improvement~N(mu,sigma)'] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)

first_year = data.index[0]
for tech_name in target_tech_names:
    print('------------------------------------')
    print(tech_name)
    print('------------------------------------')
    
    time_series = log10(data[tech_name]**-1)
    
    for training_year in training_years:
        print(training_year)
        
        start_ind = int(training_year-first_year)
        
        if sum(~isnan(time_series.values[:start_ind]))>2:            
            stan_data = {'N_time_periods': len(time_series),
                    'N_time_periods_for_inference': start_ind,
                     'Y': pd.Series(time_series).fillna(-999),
                    'mu_prior_location': .1,
                    'mu_prior_scale': 2,
                    'sigma_prior_location': 1,
                    'sigma_prior_scale': 2}

            ###
            model_fit = stanity.fit(model_code['improvement~N(mu,sigma)'], data=stan_data, n_jobs=n_jobs)
            print(_print_stanfit(model_fit, ['mu', 'sigma']))

            for horizon in horizons:
                if horizon=='all':
                    ll = nanmean(model_fit['log_likelihood'][:,start_ind:])
                else:
                    ll = nanmean(model_fit['log_likelihood'][:,start_ind:start_ind+horizon])
                technology_models_prediction['improvement~N(mu,sigma)'].ix[tech_name,horizon,training_year] = ll

            ###
            stan_data['theta_prior_location'] = .5
            stan_data['theta_prior_scale'] = 2
            
            model_fit = stanity.fit(model_code['improvement~p(theta)N(mu,sigma)'], data=stan_data, n_jobs=n_jobs)
            print(_print_stanfit(model_fit, ['mu', 'sigma', 'theta']))

            for horizon in horizons:
                if horizon=='all':
                    ll = nanmean(model_fit['log_likelihood'][:,start_ind:])
                else:
                    ll = nanmean(model_fit['log_likelihood'][:,start_ind:start_ind+horizon])
                technology_models_prediction['improvement~p(theta)N(mu,sigma)'].ix[tech_name,horizon,training_year] = ll


# In[15]:

technology_models_prediction['improvement~p(theta)N(mu,sigma)']


# In[16]:

technology_models_prediction['improvement~p(theta)N(mu,sigma)'].notnull().mean(axis=0)


# In[17]:

technology_models_prediction['improvement~p(theta)N(mu,sigma)'].mean(axis=0)


# In[18]:

technology_models_prediction['improvement~p(theta)N(mu,sigma)'].mean(axis=0).T.plot()
yscale('symlog')


# In[19]:

technology_models_prediction['improvement~N(mu,sigma)'].mean(axis=0).T.plot()
yscale('symlog')


# In[20]:

for i in arange(len(horizons)):
    figure()
    technology_models_prediction['improvement~p(theta)N(mu,sigma)'].mean(axis=0).iloc[i].plot(label='p(theta)N(mu,sigma)')
    technology_models_prediction['improvement~N(mu,sigma)'].mean(axis=0).iloc[i].plot(ax=gca(),label='N(mu,sigma)')
    yscale('symlog')
    title(str(horizons[i])+' year forecast horizon')
    xlabel("Training Model with data up to this year")
    ylabel("log(likelihood) of subsequent data within the time horizon")


# The model with improvement~N(mu,sigma) (green line) typically predicts the future better than the model with improvement~p(theta)N(mu,sigma) (blue line). The simpler model does better prediction!

# Define and test models that evaluate multiple technologies at once
# ====
# Inclues a model that does not pool the technologies at all ('improvement~N(mu,sigma), multiple') and one that uses partial pooling ('improvement~N(mu,sigma), hierarchical')
# 
# Note that in the tests both time series are actually independent, and the hierarchical model finds this.

# In[6]:

# model_code['improvement~N(mu,sigma), multiple'] = """
# functions{
#     int first_observation_ind(vector my_array){
#         int t;
#         t = 1;
#         while(my_array[t] < -900){
#           t = t+1;
#         }
#         return t;
#     }
    
#     int last_observation_ind(vector my_array, int length){
#         int last_observation;
#         last_observation = 0; 
#         for(t in 1:length){
#           if(my_array[t] > -900){
#               last_observation = t;
#           }
#         }
#         return last_observation;
#     }
    
    
#     int count_n_observations(vector my_array) {
#         int count;
#         count = 0;
#         for (t in 1:num_elements(my_array)) {
#             if(my_array[t] > -900){
#                 count = count + 1;
#             }
#         }
#         return count;
#     }
    
#     real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
#         return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
#     }
# }

# data {
#     int N_technologies;
#     int N_time_periods; // number of time periods
#     vector[N_time_periods] Y[N_technologies]; // value each time period
    
#     int N_time_periods_for_inference;
    
#     real mu_prior_location;
#     real mu_prior_scale;
    
#     real sigma_prior_location;
#     real sigma_prior_scale;
 
# }

# transformed data {
#   int first_observation[N_technologies];
#   int last_observation[N_technologies];
#   int N_observed_time_periods_for_inference[N_technologies];
#   int r_observation_offset[N_technologies];
#   int n_observations[N_technologies];
#   int r_array_offset[N_technologies];

#   for (tech in 1:N_technologies){
#       first_observation[tech] = first_observation_ind(Y[tech][1:N_time_periods_for_inference]);
#       last_observation[tech] = last_observation_ind(Y[tech][1:N_time_periods_for_inference], 
#                           N_time_periods_for_inference);

#       N_observed_time_periods_for_inference[tech] = last_observation[tech]-first_observation[tech] + 1;
#       r_observation_offset[tech] = first_observation[tech]-1;
#       n_observations[tech] = count_n_observations(Y[tech]);
#   }
#   r_array_offset[1] = 0;
#   for (tech in 2:N_technologies){
#     r_array_offset[tech] = N_observed_time_periods_for_inference[tech-1]+r_array_offset[tech-1]-1;
#   }
# }
  
# parameters {
#     real<lower = 0> mu[N_technologies];
#     real<lower = 0> sigma[N_technologies];

#     vector<lower = 0,upper = 1>[sum(N_observed_time_periods_for_inference)-
#                                 N_technologies] r_raw; // updates
# }

# transformed parameters {
#   // Identify where the first and last non-missing data points are in Y
#   vector<lower = 0>[sum(N_observed_time_periods_for_inference)-
#                                 N_technologies] r; // updates
  
#   {
#   // Dictate that the total change between each pair of observations is equal to the observed change between them
#   // This is relevant for time periods with missing data
#   int most_recent_observation;
#   for (tech in 1:N_technologies){
#       most_recent_observation = first_observation[tech];
#       for(t in first_observation[tech]+1:last_observation[tech]) {
#           if(Y[tech][t] > -900) {
#             r[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] = 
#             r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] /
#             sum(r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])]) * 
#             (Y[tech][t]-Y[tech][most_recent_observation]);
#             most_recent_observation = t;
#             }
#         }
#       }
#   }
# }

# model {
#     mu ~ normal(mu_prior_location, mu_prior_scale);
#     sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);

#     for (tech in 1:N_technologies){
#         for(t in 1:N_observed_time_periods_for_inference[tech]-1){
#             target += lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0);
#         }
#     }
# }

# generated quantities {
#     vector[N_time_periods] Y_sim[N_technologies];
#     vector[N_time_periods] log_likelihood[N_technologies];
#     //real mean_change[N_technologies];
#     //real variance_change[N_technologies];
    
#     //for (tech in 1:N_technologies){
#     //    mean_change = mean(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
#     //    variance_change = variance(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
#     //}
    
#     //Fill out data in the missing periods
#     for (tech in 1:N_technologies){
#         for(t in first_observation[tech]:last_observation[tech]) {
#           if(Y[tech][t] > -900){
#               Y_sim[tech][t] = Y[tech][t];
#           } else{
#               Y_sim[tech][t] = Y_sim[tech][t-1] + r[r_array_offset[tech]+(t-1)-r_observation_offset[tech]];
#           } 
#         }
#     }
#     {
#     real increase_size;
#     //Fill out future data points
#     for (tech in 1:N_technologies){
#         for(t in last_observation[tech]+1:N_time_periods){
#             // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
#             // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
#             increase_size = -1.0;  
#             while (increase_size<0){
#                 increase_size = normal_rng(mu[tech],sigma[tech]);
#             }
#             Y_sim[tech][t] = increase_size + Y_sim[tech][t-1];
#         }
#     }
#     }
    
#     //Fill out past data points
#     {
#     int t;
#     real increase_size;
#     for (tech in 1:N_technologies){
#         t = first_observation[tech];
#         while(t>1){
#             increase_size = -1.0;  
#             while (increase_size<0){
#                 increase_size = normal_rng(mu[tech],sigma[tech]);
#             }
#             Y_sim[tech][t-1] = Y_sim[tech][t] - increase_size;
#             t = t-1;
#         }
#     }
#     }

#     for (tech in 1:N_technologies){
#         for(t in 2:N_time_periods){
#             if(Y[tech][t] > -900){
#                     log_likelihood[tech][t] = lower_truncated_normal_lpdf(Y[tech][t]-
#                                                                         Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);
#             }
#         }
#     }
# }
# """

# model_code['improvement~N(mu,sigma), hierarchical'] = """
# functions{
#     int first_observation_ind(vector my_array){
#         int t;
#         t = 1;
#         while(my_array[t] < -900){
#           t = t+1;
#         }
#         return t;
#     }
    
#     int last_observation_ind(vector my_array, int length){
#         int last_observation;
#         last_observation = 0; 
#         for(t in 1:length){
#           if(my_array[t] > -900){
#               last_observation = t;
#           }
#         }
#         return last_observation;
#     }
    
    
#     int count_n_observations(vector my_array) {
#         int count;
#         count = 0;
#         for (t in 1:num_elements(my_array)) {
#             if(my_array[t] > -900){
#                 count = count + 1;
#             }
#         }
#         return count;
#     }
    
#     real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
#         return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
#     }
# }

# data {
#     int N_technologies;
#     int N_time_periods; // number of time periods
#     vector[N_time_periods] Y[N_technologies]; // value each time period
    
#     int N_time_periods_for_inference;
    
#     real mu_prior_location;
#     real mu_prior_scale;
    
#     real sigma_prior_location;
#     real sigma_prior_scale;
 
# }

# transformed data {
#   int first_observation[N_technologies];
#   int last_observation[N_technologies];
#   int N_observed_time_periods_for_inference[N_technologies];
#   int r_observation_offset[N_technologies];
#   int n_observations[N_technologies];
#   int r_array_offset[N_technologies];

#   for (tech in 1:N_technologies){
#       first_observation[tech] = first_observation_ind(Y[tech][1:N_time_periods_for_inference]);
#       last_observation[tech] = last_observation_ind(Y[tech][1:N_time_periods_for_inference], 
#                           N_time_periods_for_inference);

#       N_observed_time_periods_for_inference[tech] = last_observation[tech]-first_observation[tech] + 1;
#       r_observation_offset[tech] = first_observation[tech]-1;
#       n_observations[tech] = count_n_observations(Y[tech]);
#   }
#   r_array_offset[1] = 0;
#   for (tech in 2:N_technologies){
#     r_array_offset[tech] = N_observed_time_periods_for_inference[tech-1]+r_array_offset[tech-1]-1;
#   }
# }
  
# parameters {
#     vector<lower = 0>[N_technologies] mu;
#     vector<lower = 0>[N_technologies] sigma;

#     vector<lower = 0,upper = 1>[sum(N_observed_time_periods_for_inference)-
#                                 N_technologies] r_raw; // updates
                                
#     corr_matrix[2] Omega;
#     vector<lower = 0>[2] tau;
#     vector<lower=0>[2] mu_parvec;
# }

# transformed parameters {
#   // Identify where the first and last non-missing data points are in Y
#   vector<lower = 0>[sum(N_observed_time_periods_for_inference)-
#                                 N_technologies] r; // updates
#   matrix[N_technologies, 2] parvec;
#   parvec = append_col(mu, sigma);
  
#   {
#   // Dictate that the total change between each pair of observations is equal to the observed change between them
#   // This is relevant for time periods with missing data
#   int most_recent_observation;
#   for (tech in 1:N_technologies){
#       most_recent_observation = first_observation[tech];
#       for(t in first_observation[tech]+1:last_observation[tech]) {
#           if(Y[tech][t] > -900) {
#             r[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] = 
#             r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] /
#             sum(r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])]) * 
#             (Y[tech][t]-Y[tech][most_recent_observation]);
#             most_recent_observation = t;
#             }
#         }
#       }
#   }  
# }

# model {
#     tau ~ cauchy(0, 1);
#     Omega ~ lkj_corr(4);
#     mu_parvec[1] ~ student_t(3, 1, 1);
#     mu_parvec[2] ~ student_t(3, 0.5, 0.5);
    
#     for (tech in 1:N_technologies){
#         parvec[tech] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
#         for(t in 1:N_observed_time_periods_for_inference[tech]-1){
#             target += lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0);
#         }
#     }
# }

# generated quantities {
#     vector[N_time_periods] Y_sim[N_technologies];
#     vector[N_time_periods] log_likelihood[N_technologies];
#     //real mean_change[N_technologies];
#     //real variance_change[N_technologies];
    
#     //for (tech in 1:N_technologies){
#     //    mean_change = mean(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
#     //    variance_change = variance(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
#     //}
    
#     //Fill out data in the missing periods
#     for (tech in 1:N_technologies){
#         for(t in first_observation[tech]:last_observation[tech]) {
#           if(Y[tech][t] > -900){
#               Y_sim[tech][t] = Y[tech][t];
#           } else{
#               Y_sim[tech][t] = Y_sim[tech][t-1] + r[r_array_offset[tech]+(t-1)-r_observation_offset[tech]];
#           } 
#         }
#     }
#     {
#     real increase_size;
#     //Fill out future data points
#     for (tech in 1:N_technologies){
#         for(t in last_observation[tech]+1:N_time_periods){
#             // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
#             // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
#             increase_size = -1.0;  
#             while (increase_size<0){
#                 increase_size = normal_rng(mu[tech],sigma[tech]);
#             }
#             Y_sim[tech][t] = increase_size + Y_sim[tech][t-1];
#         }
#     }
#     }
    
#     //Fill out past data points
#     {
#     int t;
#     real increase_size;
#     for (tech in 1:N_technologies){
#         t = first_observation[tech];
#         while(t>1){
#             increase_size = -1.0;  
#             while (increase_size<0){
#                 increase_size = normal_rng(mu[tech],sigma[tech]);
#             }
#             Y_sim[tech][t-1] = Y_sim[tech][t] - increase_size;
#             t = t-1;
#         }
#     }
#     }

#     for (tech in 1:N_technologies){
#         for(t in 2:N_time_periods){
#             if(Y[tech][t] > -900){
#                     log_likelihood[tech][t] = lower_truncated_normal_lpdf(Y[tech][t]-
#                                                                         Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);
#             }
#         }
#     }
# }
# """


# model_code['improvement~p(theta)N(mu,sigma), hierarchical'] = """
# functions{
#     int first_observation_ind(vector my_array){
#         int t;
#         t = 1;
#         while(my_array[t] < -900){
#           t = t+1;
#         }
#         return t;
#     }
    
#     int last_observation_ind(vector my_array, int length){
#         int last_observation;
#         last_observation = 0; 
#         for(t in 1:length){
#           if(my_array[t] > -900){
#               last_observation = t;
#           }
#         }
#         return last_observation;
#     }
    
    
#     int count_n_observations(vector my_array) {
#         int count;
#         count = 0;
#         for (t in 1:num_elements(my_array)) {
#             if(my_array[t] > -900){
#                 count = count + 1;
#             }
#         }
#         return count;
#     }
    
#     real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
#         return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
#     }
# }

# data {
#     int N_technologies;
#     int N_time_periods; // number of time periods
#     vector[N_time_periods] Y[N_technologies]; // value each time period
    
#     int N_time_periods_for_inference;
    
#     real mu_prior_location;
#     real mu_prior_scale;
    
#     real sigma_prior_location;
#     real sigma_prior_scale;
    
#     real theta_prior_location;
#     real theta_prior_scale;
 
# }

# transformed data {
#   int first_observation[N_technologies];
#   int last_observation[N_technologies];
#   int N_observed_time_periods_for_inference[N_technologies];
#   int r_observation_offset[N_technologies];
#   int n_observations[N_technologies];
#   int r_array_offset[N_technologies];

#   for (tech in 1:N_technologies){
#       first_observation[tech] = first_observation_ind(Y[tech][1:N_time_periods_for_inference]);
#       last_observation[tech] = last_observation_ind(Y[tech][1:N_time_periods_for_inference], 
#                           N_time_periods_for_inference);

#       N_observed_time_periods_for_inference[tech] = last_observation[tech]-first_observation[tech] + 1;
#       r_observation_offset[tech] = first_observation[tech]-1;
#       n_observations[tech] = count_n_observations(Y[tech]);
#   }
#   r_array_offset[1] = 0;
#   for (tech in 2:N_technologies){
#     r_array_offset[tech] = N_observed_time_periods_for_inference[tech-1]+r_array_offset[tech-1]-1;
#   }
# }
  
# parameters {
#     vector<lower = 0>[N_technologies] mu;
#     vector<lower = 0>[N_technologies] sigma;
#     vector<lower = 0, upper=1>[N_technologies] theta;


#     vector<lower = 0,upper = 1>[sum(N_observed_time_periods_for_inference)-
#                                 N_technologies] r_raw; // updates
                                
#     corr_matrix[3] Omega;
#     vector<lower = 0>[3] tau;
    
#     real<lower = 0> mu_mu;
#     real<lower = 0> mu_sigma;
#     real<lower = 0, upper=1> mu_theta;
# }

# transformed parameters {
#   // Identify where the first and last non-missing data points are in Y
#   vector<lower = 0>[sum(N_observed_time_periods_for_inference)-
#                                 N_technologies] r; // updates
#   matrix[N_technologies, 3] parvec;
#   vector[3] mu_parvec;
  
#   parvec = append_col(mu, append_col(sigma,theta));
#   mu_parvec[1] = mu_mu;
#   mu_parvec[2] = mu_sigma;
#   mu_parvec[3] = mu_theta;
  
#   {
#   // Dictate that the total change between each pair of observations is equal to the observed change between them
#   // This is relevant for time periods with missing data
#   int most_recent_observation;
#   for (tech in 1:N_technologies){
#       most_recent_observation = first_observation[tech];
#       for(t in first_observation[tech]+1:last_observation[tech]) {
#           if(Y[tech][t] > -900) {
#             r[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] = 
#             r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] /
#             sum(r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
#               (r_array_offset[tech]+(t-1)-r_observation_offset[tech])]) * 
#             (Y[tech][t]-Y[tech][most_recent_observation]);
#             most_recent_observation = t;
#             }
#         }
#       }
#   }  
# }

# model {
#     tau ~ cauchy(0, 1);
#     Omega ~ lkj_corr(4);
#     mu_mu ~ normal(mu_prior_location, mu_prior_scale);
#     mu_sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
#     mu_theta ~ normal(theta_prior_location, theta_prior_scale);

#     for (tech in 1:N_technologies){
#         parvec[tech] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
#         for(t in 1:N_observed_time_periods_for_inference[tech]-1){
#             target += log_mix(theta[tech], 
#                       lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0), 
#                       lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | 0, 0.01, 0));
#         }
#     }
# }

# generated quantities {
#     vector[N_time_periods] Y_sim[N_technologies];
#     vector[N_time_periods] log_likelihood[N_technologies];
#     //real mean_change[N_technologies];
#     //real variance_change[N_technologies];
    
#     //for (tech in 1:N_technologies){
#     //    mean_change = mean(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
#     //    variance_change = variance(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
#     //}
    
#     //Fill out data in the missing periods
#     for (tech in 1:N_technologies){
#         for(t in first_observation[tech]:last_observation[tech]) {
#           if(Y[tech][t] > -900){
#               Y_sim[tech][t] = Y[tech][t];
#           } else{
#               Y_sim[tech][t] = Y_sim[tech][t-1] + r[r_array_offset[tech]+(t-1)-r_observation_offset[tech]];
#           } 
#         }
#     }
#     {
#     real increase_size;
#     //Fill out future data points
#     for (tech in 1:N_technologies){
#         for(t in last_observation[tech]+1:N_time_periods){
#             // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
#             // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
#             increase_size = -1.0;  
#             while (increase_size<0){
#                 increase_size = normal_rng(mu[tech],sigma[tech]);
#             }
#             Y_sim[tech][t] = bernoulli_rng(theta[tech])*increase_size + Y_sim[tech][t-1];
#         }
#     }
#     }
    
#     //Fill out past data points
#     {
#     int t;
#     real increase_size;
#     for (tech in 1:N_technologies){
#         t = first_observation[tech];
#         while(t>1){
#             increase_size = -1.0;  
#             while (increase_size<0){
#                 increase_size = normal_rng(mu[tech],sigma[tech]);
#             }
#             Y_sim[tech][t-1] = Y_sim[tech][t] - bernoulli_rng(theta[tech])*increase_size;
#             t = t-1;
#         }
#     }
#     }

#     for (tech in 1:N_technologies){
#         for(t in 2:N_time_periods){
#             if(Y[tech][t] > -900){                                                         
#                 if((Y[tech][t]-Y_sim[tech][t-1])==0) {
#                     log_likelihood[tech][t] = log(1-theta[tech]);
#                 } else {
#                     log_likelihood[tech][t] = log(theta[tech]) + 
#                     lower_truncated_normal_lpdf(Y[tech][t]-Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);
#                 }
#             }
#         }
#     }
# }
# """


# In[ ]:

# ### Random walk, missing data, positive steps, multiple technologies
# p_missing = 0.3
# mu = 5
# sigma = 1
# n = 100


# a = -mu / sigma
# data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

# time_series = cumsum(data)
# missing = rand(n)<p_missing
# missing[0] = False
# missing[-1] = False
# time_series[missing]=nan

# time_series0 = time_series


# p_missing = 0.2
# mu = 3
# sigma = 2
# n = 100


# a = -mu / sigma
# data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

# time_series = cumsum(data)
# missing = rand(n)<p_missing
# missing[0] = False
# missing[-1] = False
# time_series[missing]=nan

# time_series1 = time_series


# stan_data = {'N_technologies': 2,
#             'N_time_periods': n,
#             'N_time_periods_for_inference': n,
#              'Y': pd.DataFrame([time_series0, time_series1]).fillna(-999).values,
#             'mu_prior_location': 3,
#             'mu_prior_scale': 1,
#             'sigma_prior_location': 1,
#             'sigma_prior_scale': 2}

# model_fit = stanity.fit(model_code['improvement~N(mu,sigma), multiple'], data=stan_data, n_jobs=n_jobs)

# print(_print_stanfit(model_fit, ['mu', 'sigma']))

# ### Random walk, missing data, positive steps, multiple technologies, hierarchical
# p_missing = 0.3
# mu = 5
# sigma = 1
# n = 100


# a = -mu / sigma
# data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

# time_series = cumsum(data)
# missing = rand(n)<p_missing
# missing[0] = False
# missing[-1] = False
# time_series[missing]=nan

# time_series0 = time_series


# p_missing = 0.2
# mu = 3
# sigma = 2
# n = 100


# a = -mu / sigma
# data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

# time_series = cumsum(data)
# missing = rand(n)<p_missing
# missing[0] = False
# missing[-1] = False
# time_series[missing]=nan

# time_series1 = time_series


# stan_data = {'N_technologies': 2,
#             'N_time_periods': n,
#             'N_time_periods_for_inference': n,
#              'Y': pd.DataFrame([time_series0, time_series1]).fillna(-999).values,
#             'mu_prior_location': 3,
#             'mu_prior_scale': 1,
#             'sigma_prior_location': 1,
#             'sigma_prior_scale': 2}

# model_fit = stanity.fit(model_code['improvement~N(mu,sigma), hierarchical'], data=stan_data, n_jobs=n_jobs)

# print(_print_stanfit(model_fit, ['mu', 'sigma', 'Omega']))


# Fit models to empirical data and forecast the future
# ====

# In[8]:

data = pd.read_csv(data_directory+'time_series.csv',index_col=0)
data = data.reindex(arange(data.index[0],data.index[-1]+1))
metadata = pd.read_csv(data_directory+'time_series_metadata.csv')
technology_models_prediction = {}
technology_models_parameters = {}

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']
print("%i technologies"%target_tech_names.shape[0])

training_years = arange(1950,2000,10)
horizons = [5,10,'all']


# In[28]:

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']
print("%i technologies"%target_tech_names.shape[0])

training_years = arange(1950,2000,10)
horizons = [5,10,'all']

technology_models_prediction['improvement~N(mu,sigma), multiple'] = pd.Panel(items=target_tech_names,
         major_axis=horizons+['mu', 'sigma'], 
         minor_axis=training_years)

first_year = data.index[0]

time_series = data[target_tech_names]
time_series = log10(time_series**-1)

for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    time_series_from_time_period = time_series.columns[time_series.iloc[:start_ind].notnull().sum(axis=0)>2]
    n_time_series_from_time_period = len(time_series_from_time_period)
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                 'mu_prior_location': .1,
                'mu_prior_scale': 2,
                'sigma_prior_location': 1,
                'sigma_prior_scale': 2}

    ###
    model_fit = stanity.fit(model_code['improvement~N(mu,sigma), multiple'], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, ['mu', 'sigma']))
    technology_models_prediction['improvement~N(mu,sigma), multiple'].ix[time_series_from_time_period,
                                                                             'mu',training_year] = model_fit['mu'].mean(axis=0)
    technology_models_prediction['improvement~N(mu,sigma), multiple'].ix[time_series_from_time_period,
                                                                             'mu',training_year] = model_fit['sigma'].mean(axis=0)
    
    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction['improvement~N(mu,sigma), multiple'].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll


# In[29]:

target_tech_names = metadata.ix[metadata['Type']=='Performance', 'Name']
print("%i technologies"%target_tech_names.shape[0])

training_years = arange(1950,2000,10)
horizons = [5,10,'all']

technology_models_prediction['improvement~N(mu,sigma), hierarchical'] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)

first_year = data.index[0]

time_series = data[target_tech_names]
time_series = log10(time_series**-1)

for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    time_series_from_time_period = time_series.columns[time_series.iloc[:start_ind].notnull().sum(axis=0)>2]
    n_time_series_from_time_period = len(time_series_from_time_period)
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                     'mu_prior_location': .1,
                    'mu_prior_scale': 2,
                    'sigma_prior_location': 1,
                    'sigma_prior_scale': 2}

    ###
    model_fit = stanity.fit(model_code['improvement~N(mu,sigma), hierarchical'], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, ['mu', 'sigma']))

    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction['improvement~N(mu,sigma), hierarchical'].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll


# In[32]:

for i in arange(len(horizons)):
    figure()
#     technology_models_prediction['improvement~N(mu,sigma), multiple'].mean(axis=0).iloc[i].plot(ax=gca(),label='N(mu,sigma)')
    technology_models_prediction['improvement~N(mu,sigma), hierarchical'].mean(axis=0).iloc[i].plot(ax=gca(),label='N(mu,sigma), hierarchical')

    yscale('symlog')
    title(str(horizons[i])+' year forecast horizon')
    xlabel("Training Model with data up to this year")
    ylabel("log(likelihood) of subsequent data within the time horizon")


# In[33]:

for i in arange(len(time_series_from_time_period))[:10]:
    figure()
    title(time_series_from_time_period[i])
    plot_time_series_inference(model_fit,ind=i, x=data.index.values)
    scatter(time_series.index, time_series[time_series_from_time_period[i]])


# In[ ]:

model_type = 'improvement~p(theta)N(mu,sigma), hierarchical'
model_parameters = ['mu', 'sigma','theta']
parameter_priors = {'mu_prior_location': 0,
                    'mu_prior_scale': 1,
                    'sigma_prior_location': 0,
                    'sigma_prior_scale': 2,
                'theta_prior_location': .5,
                'theta_prior_scale': 2
                }

technology_models_prediction[model_type] = pd.Panel(items=target_tech_names,
         major_axis=horizons, 
         minor_axis=training_years)
technology_models_parameters[model_type] = pd.Panel(items=target_tech_names,
         major_axis=model_parameters, 
         minor_axis=training_years)


first_year = data.index[0]

time_series = data[target_tech_names]
time_series = log10(time_series**-1)

for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    ### Select only those time series that have at least 3 data points from before the training year
    time_series_from_time_period = time_series.columns[time_series.iloc[:start_ind].notnull().sum(axis=0)>=3]
    n_time_series_from_time_period = len(time_series_from_time_period)
    
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                     }
    stan_data = {**stan_data, **parameter_priors} 

    ###
    model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, model_parameters))
    for parameter in model_parameters:
        technology_models_parameters[model_type].ix[time_series_from_time_period, 
                                                    parameter,
                                                    training_year] = model_fit[parameter].mean(axis=0)
    
    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll


# In[37]:

model_code['improvement~N(mu,sigma), hierarchical, lognormal improvements'] = """
functions{
    int first_observation_ind(vector my_array){
        int t;
        t = 1;
        while(my_array[t] < -900){
          t = t+1;
        }
        return t;
    }
    
    int last_observation_ind(vector my_array, int length){
        int last_observation;
        last_observation = 0; 
        for(t in 1:length){
          if(my_array[t] > -900){
              last_observation = t;
          }
        }
        return last_observation;
    }
    
    
    int count_n_observations(vector my_array) {
        int count;
        count = 0;
        for (t in 1:num_elements(my_array)) {
            if(my_array[t] > -900){
                count = count + 1;
            }
        }
        return count;
    }
    
}

data {
    int N_technologies;
    int N_time_periods; // number of time periods
    vector[N_time_periods] Y[N_technologies]; // value each time period
    
    int N_time_periods_for_inference;
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
 
}

transformed data {
  int first_observation[N_technologies];
  int last_observation[N_technologies];
  int N_observed_time_periods_for_inference[N_technologies];
  int r_observation_offset[N_technologies];
  int n_observations[N_technologies];
  int r_array_offset[N_technologies];

  for (tech in 1:N_technologies){
      first_observation[tech] = first_observation_ind(Y[tech][1:N_time_periods_for_inference]);
      last_observation[tech] = last_observation_ind(Y[tech][1:N_time_periods_for_inference], 
                          N_time_periods_for_inference);

      N_observed_time_periods_for_inference[tech] = last_observation[tech]-first_observation[tech] + 1;
      r_observation_offset[tech] = first_observation[tech]-1;
      n_observations[tech] = count_n_observations(Y[tech]);
  }
  r_array_offset[1] = 0;
  for (tech in 2:N_technologies){
    r_array_offset[tech] = N_observed_time_periods_for_inference[tech-1]+r_array_offset[tech-1]-1;
  }
}
  
parameters {
    vector[N_technologies] mu;
    vector<lower = 0>[N_technologies] sigma;

    vector<lower = 0,upper = 1>[sum(N_observed_time_periods_for_inference)-
                                N_technologies] r_raw; // updates
                                
    corr_matrix[2] Omega;
    vector<lower = 0>[2] tau;
    vector<lower=0>[2] mu_parvec;
}

transformed parameters {
  // Identify where the first and last non-missing data points are in Y
  vector<lower = 0>[sum(N_observed_time_periods_for_inference)-
                                N_technologies] r; // updates
  matrix[N_technologies, 2] parvec;
  parvec = append_col(mu, sigma);
  
  {
  // Dictate that the total change between each pair of observations is equal to the observed change between them
  // This is relevant for time periods with missing data
  int most_recent_observation;
  for (tech in 1:N_technologies){
      most_recent_observation = first_observation[tech];
      for(t in first_observation[tech]+1:last_observation[tech]) {
          if(Y[tech][t] > -900) {
            r[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
              (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] = 
            r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
              (r_array_offset[tech]+(t-1)-r_observation_offset[tech])] /
            sum(r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):
              (r_array_offset[tech]+(t-1)-r_observation_offset[tech])]) * 
            (Y[tech][t]-Y[tech][most_recent_observation]);
            most_recent_observation = t;
            }
        }
      }
  }  
}

model {
    tau ~ cauchy(0, 1);
    Omega ~ lkj_corr(4);
    mu_parvec[1] ~ student_t(3, 1, 1);
    mu_parvec[2] ~ student_t(3, 0.5, 0.5);
    
    for (tech in 1:N_technologies){
        parvec[tech] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
        for(t in 1:N_observed_time_periods_for_inference[tech]-1){
            target += lognormal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech]);
        }
    }
}

generated quantities {
    vector[N_time_periods] Y_sim[N_technologies];
    vector[N_time_periods] log_likelihood[N_technologies];
    //real mean_change[N_technologies];
    //real variance_change[N_technologies];
    
    //for (tech in 1:N_technologies){
    //    mean_change = mean(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
    //    variance_change = variance(r[r_array_offset[tech]:(r_array_offset[tech]+N_observed_time_periods_for_inference[tech])]);
    //}
    
    //Fill out data in the missing periods
    for (tech in 1:N_technologies){
        for(t in first_observation[tech]:last_observation[tech]) {
          if(Y[tech][t] > -900){
              Y_sim[tech][t] = Y[tech][t];
          } else{
              Y_sim[tech][t] = Y_sim[tech][t-1] + r[r_array_offset[tech]+(t-1)-r_observation_offset[tech]];
          } 
        }
    }
    {
    real increase_size;
    //Fill out future data points
    for (tech in 1:N_technologies){
        for(t in last_observation[tech]+1:N_time_periods){
            // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
            // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
            increase_size = -1.0;  
            while (increase_size<0){
                increase_size = lognormal_rng(mu[tech],sigma[tech]);
            }
            Y_sim[tech][t] = increase_size + Y_sim[tech][t-1];
        }
    }
    }
    
    //Fill out past data points
    {
    int t;
    real increase_size;
    for (tech in 1:N_technologies){
        t = first_observation[tech];
        while(t>1){
            increase_size = -1.0;  
            while (increase_size<0){
                increase_size = lognormal_rng(mu[tech],sigma[tech]);
            }
            Y_sim[tech][t-1] = Y_sim[tech][t] - increase_size;
            t = t-1;
        }
    }
    }

    for (tech in 1:N_technologies){
        for(t in 2:N_time_periods){
            if(Y[tech][t] > -900){
                    log_likelihood[tech][t] = lognormal_lpdf(Y[tech][t]-
                                                                        Y_sim[tech][t-1]| mu[tech], sigma[tech]);
            }
        }
    }
}
"""


# In[38]:

model_type = 'improvement~N(mu,sigma), hierarchical, lognormal improvements'


technology_models_prediction[model_type] = pd.Panel(items=target_tech_names,
         major_axis=horizons+['mu', 'sigma'], 
         minor_axis=training_years)

first_year = data.index[0]

time_series = data[target_tech_names]
time_series = log10(time_series**-1)

for training_year in training_years:
    print(training_year)

    start_ind = int(training_year-first_year)
    time_series_from_time_period = time_series.columns[time_series.iloc[:start_ind].notnull().sum(axis=0)>2]
    n_time_series_from_time_period = len(time_series_from_time_period)
    stan_data = {'N_technologies': n_time_series_from_time_period,
                 'N_time_periods': time_series.shape[0],
            'N_time_periods_for_inference': start_ind,
             'Y': time_series[time_series_from_time_period].fillna(-999).T,
                     'mu_prior_location': log(.1),
                    'mu_prior_scale': log(2),
                    'sigma_prior_location': log(1),
                    'sigma_prior_scale': log(2)}

    ###
    model_fit = stanity.fit(model_code[model_type], data=stan_data, n_jobs=n_jobs)
    print(_print_stanfit(model_fit, ['mu', 'sigma']))
    technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             'mu',training_year] = model_fit['mu'].mean(axis=0)
    technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             'mu',training_year] = model_fit['sigma'].mean(axis=0)
    
    for horizon in horizons:
        if horizon=='all':
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:],axis=0),axis=1)
        else:
            ll = nanmean(nanmean(model_fit['log_likelihood'][:,:,start_ind:start_ind+horizon],axis=0),axis=1)
        technology_models_prediction[model_type].ix[time_series_from_time_period,
                                                                             horizon,training_year] = ll

