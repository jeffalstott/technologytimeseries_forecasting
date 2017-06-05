
# coding: utf-8

# In[1]:

# import pystan
import stanity
n_jobs = 4
import pandas as pd
import seaborn
get_ipython().magic('pylab inline')


# In[2]:

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
        header = "Inference for Stan model: {}.\n".format(fit.model_name)
        header += "{} chains, each with iter={}; warmup={}; thin={}; \n"
        header = header.format(fit.sim['chains'], fit.sim['iter'], fit.sim['warmup'],
                               fit.sim['thin'], sum(n_kept))
        header += "post-warmup draws per chain={}, total post-warmup draws={}.\n\n"
        header = header.format(n_kept[0], sum(n_kept))
        footer = "\n\nSamples were drawn using {} at {}.\n"            "For each parameter, n_eff is a crude measure of effective sample size,\n"            "and Rhat is the potential scale reduction factor on split chains (at \n"            "convergence, Rhat=1)."
        sampler = fit.sim['samples'][0]['args']['sampler_t']
        date = fit.date.strftime('%c')  # %c is locale's representation
        footer = footer.format(sampler, date)
        s = _summary(fit, pars, probs)
        body = _array_to_table(s['summary'], s['summary_rownames'],
                               s['summary_colnames'], digits_summary)
        return header + body + footer


# In[3]:

def plot_time_series_inference(model_fit, var='Y_sim', x=None,
                               ax=None):
    from scipy.stats import scoreatpercentile
    ci_thresholds = [2.5, 25, 75, 97.5]
    CIs = scoreatpercentile(model_fit[var], ci_thresholds, axis=0)
    CIs = pd.DataFrame(data=CIs.T, columns=ci_thresholds)
    if ax is None:
        ax=gca()
    if x is None:
        x = arange(model_fit['Y_sim'].shape[1])
    ax.fill_between(x, CIs[2.5], CIs[97.5],alpha=.5)
    ax.fill_between(x, CIs[25], CIs[75])


# In[4]:

from scipy.stats import norm, truncnorm


# In[5]:

#### Random walk
mu = 0
sigma = 1
n = 1000

data = norm(loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
data {
    int T; // number of time periods
    vector[T] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real mu;
    real<lower = 0> sigma;
    
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 1:(T-1)) {
        Y[t+1] ~ normal(mu+Y[t],sigma);
    }

}
"""

stan_data = {'T': len(time_series),
       'Y': time_series,
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)
print(model_fit)
model_fit.plot()
tight_layout()


# In[5]:

#### Random walk, positive steps
mu = 4
sigma = 2
n = 1000


a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
  
data {
    int T; // number of time periods
    vector[T] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower=0> mu;
    real<lower = 0> sigma;
    
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 1:(T-1)) {
        target += normal_lpdf( Y[t+1]-Y[t] | mu, sigma);
        target += -normal_lccdf(0 | mu, sigma);
    }

}
"""

stan_data = {'T': len(time_series),
       'Y': time_series,
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)
print(model_fit)
model_fit.plot()
tight_layout()


# In[18]:

### Random walk, possibility of no step

theta = .8
mu = 5
sigma = 2
n = 1000

improvement = rand(n)<theta

data = norm(loc=mu,scale=sigma).rvs(n)
data[~improvement]=0

time_series = cumsum(data)
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
  
data {
    int T; // number of time periods
    vector[T] Y; // value each time period
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real mu;
    real<lower = 0> sigma;
    real<lower=0, upper=1> theta;
    
}

model {
    theta ~ normal(theta_prior_location, theta_prior_scale);

    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 1:(T-1)) {
        target += log_mix(theta, normal_lpdf(Y[t+1]-Y[t] | mu, sigma), 
                      normal_lpdf(Y[t+1]-Y[t] | 0, 0.01));
    }

}
"""

stan_data = {'T': len(time_series),
       'Y': time_series,
       'theta_prior_location': .5,
       'theta_prior_scale': 1,
       'mu_prior_location': 0,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)
print(model_fit)
model_fit.plot()
tight_layout()


# In[6]:

### Random walk, possibility of no step, positive steps, noise on no step

theta = .8
mu = 5
sigma = 1
n = 1000

improvement = rand(n)<theta

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)
data[~improvement]=0

time_series = cumsum(data)
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    //real out;
    //if(x<=A) {
    //  out = log(0);
    //} else {
    //  out = normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma);
    //}
    //return(out);
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
  }
}

data {
    int T; // number of time periods
    vector[T] Y; // value each time period
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower=0> mu;
    real<lower=0> sigma;
    real<lower=0, upper=1> theta;
    
}

model {
    theta ~ normal(theta_prior_location, theta_prior_scale);

    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 1:(T-1)) {
        target += log_mix(theta, lower_truncated_normal_lpdf(Y[t+1]-Y[t] | mu, sigma, 0), 
                      lower_truncated_normal_lpdf(Y[t+1]-Y[t] | 0, 0.01, 0));
    }

}
"""

stan_data = {'T': len(time_series),
       'Y': time_series,
       'theta_prior_location': .5,
       'theta_prior_scale': 1,
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)
print(model_fit)
model_fit.plot()
tight_layout()


# In[7]:

### Random walk, possibility of no step, positive steps, no noise on no step

theta = .8
mu = 5
sigma = 1
n = 1000

improvement = rand(n)<theta

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)
data[~improvement]=0

time_series = cumsum(data)
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    //real out;
    //if(x<=A) {
    //  out = log(0);
    //} else {
    //  out = normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma);
    //}
    //return(out);
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
  }
}

data {
    int T; // number of time periods
    vector[T] Y; // value each time period
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower=0> mu;
    real<lower=0> sigma;
    real<lower=0, upper=1> theta;
    
}

model {
    theta ~ normal(theta_prior_location, theta_prior_scale);

    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 1:(T-1)) {
        if((Y[t+1]-Y[t])==0) {
          target += log(1-theta);
        } else {
          target += lower_truncated_normal_lpdf(Y[t+1]-Y[t] | mu, sigma, 0) + log(theta);
          }
    }

}
"""

stan_data = {'T': len(time_series),
       'Y': time_series,
       'theta_prior_location': .5,
       'theta_prior_scale': 1,
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)
print(model_fit)
model_fit.plot()
tight_layout()


# In[104]:

### Random walk, missing data, with measurement error

p_missing = .4
mu = 0
sigma = 1
n = 100

data = norm(loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real mu;
    real<lower = 0> sigma;
    vector[N-1] r; // updates

}

transformed parameters {
  vector[N] latent_Y;
  
  latent_Y[1] = Y[1];
  
  for(t in 2:N) {
    latent_Y[t] = latent_Y[t-1] + r[t-1];
  }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 2:N) {
        r[t-1] ~ normal(mu,sigma);
        if(Y[t] > -900) {
            Y[t] ~ normal(latent_Y[t], 0.01);
            //target += normal_lpdf( latent_Y[t]| Y[t], 0.01); // measurement error
            //latent_Y[t] ~ normal(Y[t], 0.001) T[0,]
        }
    }

}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma']))

n_T = 100
plot(model_fit['latent_Y'][:50,:n_T].T,linewidth=.1, color='gray', alpha=.05)
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0,xmax=n_T)


# In[40]:

### Random walk, missing data, positive steps, with measurement error
### Doesn't work!

p_missing = 0
mu = 4
sigma = 2
n = 1000

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    vector<lower = 0>[N-1] r; // updates

}

transformed parameters {
  vector[N] latent_Y;
  
  latent_Y[1] = Y[1];
  
  for(t in 2:N) {
    latent_Y[t] = latent_Y[t-1] + r[t-1];
  }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 2:N) {
        r[t-1] ~ normal(mu,sigma) T[0,];
        if(Y[t] > -900) {
            latent_Y[t] ~ normal(Y[t], 0.001) T[Y[t],];
            //target += normal_lpdf( latent_Y[t]| Y[t], 0.01); // measurement error
            //latent_Y[t] ~ normal(Y[t], 0.001) T[0,]
        }
    }

}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs, init_r = 5)

print(_print_stanfit(model_fit, ['mu', 'sigma']))

figure()
n_T = 100
plot(model_fit['latent_Y'][:50,:n_T].T,linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[6]:

### Random walk, missing data, positive steps, no measurement error, possibility of no step, noise on no step
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

model_code = """
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
    
    real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
        return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
    }
}

data {
    int N_time_periods; // number of time periods
    vector[N_time_periods] Y; // value each time period
    
    int N_time_periods_for_inference;
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real theta_prior_location;
    real theta_prior_scale;
}

transformed data {
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

}
  
parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    real<lower = 0, upper = 1> theta;

    vector<lower = 0,upper = 1>[N_observed_time_periods_for_inference-1] r_raw; // updates
}

transformed parameters {
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
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    theta ~ normal(theta_prior_location, theta_prior_scale);

    for(t in 1:N_observed_time_periods_for_inference-1){
        target += log_mix(theta, lower_truncated_normal_lpdf(r[t] | mu, sigma, 0), 
                      lower_truncated_normal_lpdf(r[t] | 0, 0.01, 0));
    }
}

generated quantities {
    vector[N_time_periods] Y_sim;
    vector[N_time_periods] log_lik;

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
            increase_size = normal_rng(mu,sigma);
        }
        Y_sim[t] = bernoulli_rng(theta) * increase_size + Y_sim[t-1];
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
            increase_size = normal_rng(mu,sigma);
        }
        Y_sim[t-1] = Y_sim[t] - bernoulli_rng(theta) * increase_size;
        t = t-1;
    }
    }
    
    for(t in 2:N_time_periods){
        if(Y[t] > -900){
            if((Y[t]-Y_sim[t-1])==0) {
                log_lik[t] = log(1-theta);
            } else {
                log_lik[t] = log(theta) + 
                            lower_truncated_normal_lpdf(Y[t]-Y_sim[t-1]| mu, sigma, 0);
            }
        }
    }
}
"""

n_past_steps = 25
n_future_steps = 25
stan_data = {'N_time_periods': len(time_series)+n_future_steps+n_past_steps,
            'N_time_periods_for_inference': len(time_series)+n_past_steps,
             'Y': pd.Series(concatenate((empty(n_past_steps)*nan,
                                         time_series,
                                         empty(n_future_steps)*nan),0)).fillna(-999),
           'theta_prior_location': .8,
           'theta_prior_scale': .2,
            'mu_prior_location': 3,
            'mu_prior_scale': 1,
            'sigma_prior_location': 0,
            'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma','theta']))

figure()
plot_time_series_inference(model_fit)
scatter(arange(len(time_series))+n_past_steps, time_series,s=2)
xlim(xmin=0)
# ylim(ymin=0)


# In[67]:

### Random walk, missing data, positive steps, no measurement error
p_missing = 0.3
mu = 4
sigma = 2
n = 100

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    vector<lower = 0,upper = 1>[N-1] r_raw; // updates
}

transformed parameters {
  vector<lower = 0>[N-1] r; // updates

  {
  int first_observation;
  int last_observation;
  int length_observed_time_series;
  int t;
  first_observation = 0;
  last_observation = 0;
  t = 1;
  while(first_observation=0){
      if(Y[t] > -900){
          first_observation = t;
      } else{
          t = t+1;
      }
  }
  
  t=1;
  for(t in first_observation:N){
      if(Y[t] > -900){
          last_observation = t;
  }
  
  int last_observation;
  last_observation = 1;
  for(t in 2:N) {
      if(Y[t] > -900) {
        r[last_observation:t-1] = r_raw[last_observation:t-1]/sum(r_raw[last_observation:t-1]) * (Y[t]-Y[last_observation]);
        last_observation = t;
        }
    }
    }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    for(t in 1:N-1){
        r[t] ~ normal(mu,sigma) T[0,];
    }
}

generated quantities {
  vector[N] latent_Y;
  
  latent_Y[1] = Y[1];
  
  for(t in 2:N) {
    latent_Y[t] = latent_Y[t-1] + r[t-1];
  }
}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma']))

figure()
n_T = 30
plot(arange(1,n_T+1),cumsum(model_fit['r'][:,:n_T].T, axis=0)+time_series[0],linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[5]:

### Random walk, missing data, positive steps, no measurement error, possibility of no step, noise on no step
p_missing = 0.1
theta = .8
mu = 5
sigma = 1
n = 1000

improvement = rand(n)<theta

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

model_code = """
functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
  }
}

data {
    int N; // number of time periods
    vector[N] Y; // value each time period
        
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
}

parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    vector<lower = 0,upper = 1>[N-1] r_raw; // updates
    real<lower = 0, upper = 1> theta;
}

transformed parameters {
  vector<lower = 0>[N-1] r; // updates
  
  {
  int last_observation;
  last_observation = 1;
  for(t in 2:N) {
      if(Y[t] > -900) {
        r[last_observation:t-1] = r_raw[last_observation:t-1]/sum(r_raw[last_observation:t-1]) * (Y[t]-Y[last_observation]);
        last_observation = t;
        }
    }
    }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    theta ~ normal(theta_prior_location, theta_prior_scale);

    for(t in 1:N-1){
        target += log_mix(theta, lower_truncated_normal_lpdf(r[t] | mu, sigma, 0), 
                      lower_truncated_normal_lpdf(r[t] | 0, 0.01, 0));
    }
}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'theta_prior_location': .8,
       'theta_prior_scale': .2,
       'mu_prior_location': 5,
       'mu_prior_scale': .2,
       'sigma_prior_location': 1,
       'sigma_prior_scale': 2}
model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma', 'theta']))

figure()
n_T = 30
plot(arange(1,n_T+1),cumsum(model_fit['r'][:,:n_T].T, axis=0)+time_series[0],linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[14]:

### Random walk, missing data, positive steps, no measurement error, possibility of no step, generate predictions
p_missing = 0.3
theta = .8
mu = 5
sigma = 1
n = 100
M = 100

improvement = rand(n)<theta

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

model_code = """
functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
  }
}

data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    int M; // number of future time periods to simulate
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
}

parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    vector<lower = 0,upper = 1>[N-1] r_raw; // updates
    real<lower = 0, upper = 1> theta;
}

transformed parameters {
  vector<lower = 0>[N-1] r; // updates
  
  {
  int last_observation;
  last_observation = 1;
  for(t in 2:N) {
      if(Y[t] > -900) {
        r[last_observation:t-1] = r_raw[last_observation:t-1]/sum(r_raw[last_observation:t-1]) * (Y[t]-Y[last_observation]);
        last_observation = t;
        }
    }
    }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    theta ~ normal(theta_prior_location, theta_prior_scale);

    for(t in 1:N-1){
        target += log_mix(theta, lower_truncated_normal_lpdf(r[t] | mu, sigma, 0), 
                      lower_truncated_normal_lpdf(r[t] | 0, 0.01, 0));
    }
}

generated quantities {
    vector[M] Y_sim;
    real last_observation;
    real increase_size;

    last_observation = Y[N];
    
    for(t in 1:M){
        // Stan cannot yet generate numbers from a truncated distribution directly, so we have to do this silly thing. 
        // As of version 2.12.0, the devs are still talking about it: https://github.com/stan-dev/math/issues/214
        increase_size = -1.0;  
        while (increase_size<0){
            increase_size = normal_rng(mu,sigma);
        }
        Y_sim[t] = bernoulli_rng(theta) * increase_size + last_observation;
        last_observation = Y_sim[t];
    }
}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
             'M': M,
       'theta_prior_location': .8,
       'theta_prior_scale': .2,
       'mu_prior_location': 5,
       'mu_prior_scale': .2,
       'sigma_prior_location': 1,
       'sigma_prior_scale': 2}
model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma', 'theta']))

figure()
n_T = 30
plot(arange(1,n_T+1),cumsum(model_fit['r'][:,:n_T].T, axis=0)+time_series[0],linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[27]:

from scipy.stats import scoreatpercentile
ci_thresholds = [2.5, 25, 75, 97.5]
CIs = scoreatpercentile(model_fit['Y_sim'], ci_thresholds, axis=0)
CIs = pd.DataFrame(data=CIs.T, columns=ci_thresholds)
fill_between(arange(M)+len(time_series), CIs[2.5], CIs[97.5],alpha=.5)
fill_between(arange(M)+len(time_series), CIs[25], CIs[75])
plot(time_series)


# In[69]:

### Random walk, missing data, positive steps, no measurement error, possibility of no step
p_missing = .3
theta = .8
mu = 5
sigma = 1
n = 100

improvement = rand(n)<theta

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)
data[~improvement]=0

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan


hist(data, bins=50)
figure()
plot(time_series)

model_code = """
functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
  }
}

data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
}

parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    vector<lower = 0,upper = 1>[N-1] r_raw; // updates
    real<lower = 0, upper = 1> theta;
}

transformed parameters {
  vector<lower = 0>[N-1] r; // updates

  {
  int last_observation;
  last_observation = 1;
  for(t in 2:N) {
      if(Y[t] > -900) {
        r[last_observation:t-1] = r_raw[last_observation:t-1]/sum(r_raw[last_observation:t-1]) * (Y[t]-Y[last_observation]);
        last_observation = t;
        }
    }
    }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    theta ~ normal(theta_prior_location, theta_prior_scale);

    for(t in 1:N-1){
        target += log_sum_exp(log(theta)+lower_truncated_normal_lpdf(r[t] | mu, sigma, 0), 
                             log(1-theta));
    }
}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'theta_prior_location': .5,
       'theta_prior_scale': 1,
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}
model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma', 'theta']))

figure()
n_T = 30
plot(arange(1,n_T+1),cumsum(model_fit['r'][:,:n_T].T, axis=0)+time_series[0],linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[16]:

### Random walk, missing data, positive steps, no measurement error
p_missing = 0.3
mu = 4
sigma = 2
n = 1000

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower = 0> r[N-1]; // updates
    real<lower = 0> mu;
    real<lower = 0> sigma;
}

model {
  mu ~ normal(mu_prior_location, mu_prior_scale);
  sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
  int last_observation;
  last_observation = 1;
  for(t in 2:N) {
      if(Y[t] > -900) {
        real r_raw[t-last_observation];
        sum(r[last_observation:t]) ~ = r_raw/sum(r_raw) * (Y[t]-Y[last_observation]);
        last_observation = t;
        }
    }
    r ~ normal(mu,sigma) T[0,];
}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma']))

figure()
n_T = 100
plot(model_fit['latent_Y'][:50,:n_T].T,linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[123]:

### Random walk, missing data, positive steps, no measurement error
p_missing = 0.3
mu = 4
sigma = 2
n = 1000

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan
hist(data, bins=50)
figure()
plot(time_series)

model_code = """
data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower = 0> mu;
    real<lower = 0> sigma;
    vector<lower = 0>[N-1] r; // updates

}

transformed parameters {
  vector[N] latent_Y;
  
  latent_Y[1] = Y[1];
  
  for(t in 2:N) {
    latent_Y[t] = latent_Y[t-1] + r[t-1];
  }
  {
  int last_observation;
  last_observation = 1;
  for(t in 2:N) {
      if(Y[t] > -900) {
        latent_Y[t] = Y[t];
        sum(r[last_observation:t]) = Y[t] - Y[last_observation];  \\This could be handled through an array of simplexes. Seems silly, though.
        last_observation = t;
        }
    }
    }
}

model {
    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    for(t in 2:N) {
        r[t-1] ~ normal(mu,sigma) T[0,];
        //if(Y[t] > -900) {
          //  Y[t] ~ normal(latent_Y[t], 0.001);
            //target += normal_lpdf( latent_Y[t]| Y[t], 0.01); // measurement error
            //latent_Y[t] ~ normal(Y[t], 0.001) T[0,]
        //}
    }

}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)

print(_print_stanfit(model_fit, ['mu', 'sigma']))

figure()
n_T = 100
plot(model_fit['latent_Y'][:50,:n_T].T,linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
xlim(xmin=0)


# In[44]:

### Random walk, possibility of no step, positive steps, noise on no step, missing data

p_missing = 0
theta = .8
mu = 5
sigma = 1
n = 1000

improvement = rand(n)<theta

a = -mu / sigma
data = truncnorm(a, inf, loc=mu,scale=sigma).rvs(n)
data[~improvement]=0

time_series = cumsum(data)
missing = rand(n)<p_missing
missing[0] = False
time_series[missing]=nan


hist(data, bins=50)
figure()
plot(time_series)

model_code = """
functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
  }
}

data {
    int N; // number of time periods
    vector[N] Y; // value each time period
    
    real theta_prior_location; 
    real theta_prior_scale; 
    
    real mu_prior_location;
    real mu_prior_scale;
    
    real sigma_prior_location;
    real sigma_prior_scale;
}

parameters {
    real<lower=0> mu;
    real<lower=0> sigma;
    real<lower=0, upper=1> theta;
    
    vector<lower = 0>[N-1] r; // updates

    
}

transformed parameters {
  vector[N] latent_Y;
  
  latent_Y[1] = Y[1];
  
  for(t in 2:N) {
    latent_Y[t] = latent_Y[t-1] + r[t-1];
  }
}

model {
    theta ~ normal(theta_prior_location, theta_prior_scale);

    mu ~ normal(mu_prior_location, mu_prior_scale);
    sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
    
    
    for(t in 2:N) {
        target += log_mix(theta, lower_truncated_normal_lpdf(r[t-1] | mu, sigma, 0.0), 
                      lower_truncated_normal_lpdf(r[t-1] | 0.0, 0.01, 0.0));
                      
        if(Y[t] > -900) {
            target += normal_lpdf( latent_Y[t]| Y[t], 0.01); // measurement error
            //latent_Y[t] ~ normal(Y[t], 0.001) T[0,]
        }
    }

}
"""

stan_data = {'N': len(time_series),
       'Y': pd.Series(time_series).fillna(-999),
       'theta_prior_location': .5,
       'theta_prior_scale': 1,
       'mu_prior_location': 3,
       'mu_prior_scale': 1,
       'sigma_prior_location': 0,
       'sigma_prior_scale': 2}

model_fit = stanity.fit(model_code, data=stan_data, n_jobs=n_jobs)


# In[38]:

n_T = 20
plot(model_fit['latent_Y'][:50,:n_T].T,linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
ylim(ymin=0)
xlim(xmin=0)


# In[27]:

n_T = 20
plot(model_fit['latent_Y'][:50,:n_T].T,linewidth=.5, color='gray')
scatter(arange(n_T), time_series[:n_T])
ylim(ymin=0)
xlim(xmin=0)


# In[ ]:

print(model_fit)
model_fit.plot()
tight_layout()

figure()
model_fit.extract('latent_Y').plot(linewidth=.5, color='gray')
scatter(time_series)

