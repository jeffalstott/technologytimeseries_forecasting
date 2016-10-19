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
  real<lower = 0> sigma; // scale of update
  real mu;
  vector<lower = 0>[T-1] r; // updates
}
transformed parameters {
  vector[T] latent_Y;
  latent_Y[1] = Y[1];
  for(t in 2:T) {
      latent_Y[t] =  latent_Y[t-1] + r[t-1];
  }
}
model {
  theta ~ beta(1.2, 1.2);
  mu ~ normal(.2, .1);
  sigma ~ normal(.3, .1);
  r ~ normal(mu, sigma);
  
  for(t in 2:T) {
    real mu_1;
    real mu_2;
    
    if(Y[t-1]>-900) {
      mu_1 = Y[t-1];
      mu_2 = Y[t-1] + r[t-1];
    } else {
      mu_1 = latent_Y[t-1];
      mu_2 = latent_Y[t-1] + r[t-1];
    }
    
    target += log_mix(theta, normal_lpdf(latent_Y[t] |  mu_1, 0.01), 
                      normal_lpdf(latent_Y[t] | mu_2, 0.01));
  }
  
  //Y ~ normal(latent_Y, 0.01);
  
}
