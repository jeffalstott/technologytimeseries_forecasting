
# coding: utf-8

# In[1]:

# import pystan
import stanity
n_jobs = 4
import pandas as pd
import seaborn
get_ipython().magic('pylab inline')


# In[2]:

from scipy.stats import norm, truncnorm


# In[4]:

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


# In[37]:

### Random walk, possibility of no step, positive steps, noise on no step, missing data

p_missing = .3
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


# In[41]:

print(model_fit)


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

