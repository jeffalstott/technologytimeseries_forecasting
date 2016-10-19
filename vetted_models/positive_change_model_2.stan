data {
  int T; // number of time periods
  vector[T] Y;
}
parameters {
  real<lower = 0, upper = 1> theta; // probability of update in a given year
  real<lower = 0> sigma; // scale of update
  real<lower = 0> mu;
  vector<lower = 0>[T-1] r; // updates
}
transformed parameters {
  vector[T] latent_Y;
  
  latent_Y[1] = Y[1];
  
  for(t in 2:T) {
    latent_Y[t] = latent_Y[t-1] + r[t-1];
  }
  
}
/*model {
  theta ~ beta(2, 2);
  sigma ~ cauchy(0, 2);
  mu ~ normal(0, 1);
  for(t in 1:(T-1)) {
    target += log_sum_exp(log((1 - theta)) + normal_lpdf(r[t] | 0, 0.01), 
                          log(theta) + normal_lpdf(r[t] | mu, sigma));
                          

  }
}*/
model {
  sigma ~ cauchy(0, 2);
  mu ~ normal(0, 1);
  theta ~ beta(1, 1);
  
  for(t in 2:T) {
    if(r[t-1]==0) {
      target += log_mix(theta, normal_lpdf(r[t-1] | mu, sigma), 0);
    } else {
      target += normal_lpdf(r[t-1] | mu, sigma);
    }
    
    // target += log_sum_exp(log(theta) + normal_lpdf(latent_Y[t] | mu_1, 0.01), log(1-theta) + normal_lpdf(latent_Y[t] | mu_2, 0.01));
        // measurement model                      
   if(Y[t] != -999.0) {
     target += normal_lpdf( latent_Y[t]| Y[t], 0.01);
   }
  }

}