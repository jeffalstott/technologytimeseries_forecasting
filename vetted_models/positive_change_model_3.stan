data {
  int T; // number of time periods
  vector[T] Y;
}
transformed data {
  vector[T-1] r;
  for(t in 1:(T-1)) {
    r[t] = Y[t+1] - Y[t];
  }
}
parameters {
  real<lower = 0, upper = 1> theta; // probability of update in a given year
  real<lower = 0> sigma; // scale of update
  real<lower = 0> mu;
}
model {
  sigma ~ cauchy(0, 2);
  mu ~ normal(0, 1);
  theta ~ beta(1, 1);
  
  for(t in 2:T) {
    if(r[t-1]==0) {
      target += log(1-theta);
    } else {
      target += lognormal_lpdf(r[t-1] | mu, sigma) + log(theta);
      }
    }
}