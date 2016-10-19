functions {
  real lower_truncated_normal_lpdf(real x, real mu, real sigma, real A) {
    real out;
    if(x<=A) {
      out = log(0);
    } else {
      out = -log(sigma) + normal_lpdf(x| mu, sigma) - log(1 - normal_cdf(A, mu, sigma));
    }
    return(out);
  }
}
data {
  int T; // number of time periods
  vector[T] Y;
}
parameters {
  real<lower = 0, upper = 1> theta; // probability of update in a given year
  vector<lower = 0>[T] me;
  real<lower = 0> sigma; // scale of update
  real mu;
  vector<lower = 0>[T-1] r; // updates
}
transformed parameters {
  vector[T] latent_Y;
  
  latent_Y[1] = Y[1] + me[1];
  
  for(t in 2:T) {
    if(Y[t-1]>-900) {
      latent_Y[t] = Y[t-1] + me[t-1] + r[t-1];
    } else {
      latent_Y[t] =  latent_Y[t-1] + r[t-1];
    }
    
  }
  
}
model {
  real ytemp;
  sigma ~ normal(.3, .1);
  mu ~ normal(0.2, .1);
  theta ~ beta(1.2, 1.2);
  
  for(t in 2:T) {
    if(fabs(latent_Y[t] - latent_Y[t-1])<me[t]) {
      //target += log(1-theta) + normal_lpdf(r[t-1] | 0, 0.01);
      target += log_mix(1-theta, normal_lpdf(r[t-1] | 0, 0.01),
                        lognormal_lpdf(r[t-1] | mu, sigma));
      //target += log(1-theta);
      //target += log_mix(theta, normal_lpdf(r[t-1] | mu, sigma), 0);
      //target += log_mix(theta, lower_truncated_normal_lpdf(r[t-1] | mu, sigma, 0.0), 0.0);
    } else {
      //target += lower_truncated_normal_lpdf(r[t-1] | mu, sigma, 0.0) + log(theta);
      target += lognormal_lpdf(r[t-1] | mu, sigma) + log(theta);
    }
  }
  
  
  me ~ normal(0, 0.01);

}
