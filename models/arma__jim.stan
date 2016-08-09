data { 
  int<lower=1> T; 
  real y[T];
}

parameters { 
  real mu; 
  real<lower=-1,upper=1> theta; 
  real<lower=0> sigma; 
} 
model {
  vector[T] nu;
  vector[T] err;
  nu[1] = mu +  mu;
  err[1] = y[1] - nu[1];
  for (t in 2:T) {
    nu[t] = mu  + theta * err[t-1];
    err[t] = y[t] - nu[t];
  }
  mu ~ normal(0, 1);
  theta ~ normal(0, 1);
  sigma ~ student_t(4, 0, 2);
  err ~ normal(0, sigma);

}
