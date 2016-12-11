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

real lower_truncated_normal_lpdf_vector(vector x, real mu, real sigma, real A) {
    return(normal_lpdf(x | mu, sigma) - normal_lccdf(A | mu, sigma));
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

vector<lower = 0, upper = 1>[sum(N_observed_time_periods_for_inference)-N_technologies] r_raw; // updates

vector<lower = 0>[N_technologies] mu;
vector<lower = 0>[N_technologies] sigma;
                
                //corr_matrix[2] Omega;
cholesky_factor_corr[2] L_Omega;
vector<lower = 0>[2] tau;
                //vector<lower = 0>[2] z;
                
real<lower = 0> mu_mu;
real<lower = 0> mu_sigma;
                

}

transformed parameters {

// Identify where the first and last non-missing data points are in Y
vector<lower = 0>[sum(N_observed_time_periods_for_inference)-
                            N_technologies] r; // updates

{
// Dictate that the total change between each pair of observations is equal to the observed change between them
// This is relevant for time periods with missing data
int most_recent_observation;
for (tech in 1:N_technologies){
  most_recent_observation = first_observation[tech];
  for(t in (first_observation[tech]+1):last_observation[tech]) {
      if(Y[tech][t] > -900) {
        r[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):(r_array_offset[tech]+(t-1)-r_observation_offset[tech])] =
        r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):(r_array_offset[tech]+(t-1)-r_observation_offset[tech])] /
        sum(r_raw[(r_array_offset[tech]+most_recent_observation-r_observation_offset[tech]):(r_array_offset[tech]+(t-1)-r_observation_offset[tech])]) * 
        (Y[tech][t]-Y[tech][most_recent_observation]);
        most_recent_observation = t;
        }
    }
  }
}

}

model {

            tau ~ cauchy(0, 1);
            L_Omega ~ lkj_corr_cholesky(3);
            mu_mu ~ lognormal(mu_prior_location, mu_prior_scale);
            mu_sigma ~ cauchy(sigma_prior_location, sigma_prior_scale);
            
            {
            matrix[N_technologies, 2] parvec;
            vector[2] mu_parvec;

            parvec = append_col(mu, sigma);
            mu_parvec[1] = mu_mu;
            mu_parvec[2] = mu_sigma;
            
            for (tech in 1:N_technologies){
                log(parvec[tech]) ~ multi_normal_cholesky(log(mu_parvec), diag_pre_multiply(tau, L_Omega));
                
                for(t in 1:N_observed_time_periods_for_inference[tech]-1){
                    target += lower_truncated_normal_lpdf(r[r_array_offset[tech]+t] | mu[tech], sigma[tech], 0);
                }
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
            increase_size = normal_rng(mu[tech],sigma[tech]);
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
            increase_size = normal_rng(mu[tech],sigma[tech]);
        }
        Y_sim[tech][t-1] = Y_sim[tech][t] - increase_size;
        t = t-1;
    }
}
}

for (tech in 1:N_technologies){
  log_likelihood[tech][1] = 0;
    for(t in 2:N_time_periods){
        if(Y[tech][t] > -900){
                
                    log_likelihood[tech][t] = lower_truncated_normal_lpdf(Y[tech][t]-
                                                                    Y_sim[tech][t-1]| mu[tech], sigma[tech], 0);;
                
        } else {
          log_likelihood[tech][t] = 0;
        }
    }
}

}
