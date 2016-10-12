data {
  int T;
  vector[T] y;
}
parameters {
  vector[T] state;
  real state_0;
  real phi;
  real theta;
  real<lower = 0> sigma;
  real mu;
}
transformed parameters{
  real K;
  K <- sqrt( (1+theta^2) * sigma^2 );
}
model {
  real err;
  // priors
  phi ~ normal(1, .1);
  theta ~ normal(.5, .5);
  mu ~ normal(0, 1);
  sigma ~ cauchy(0, 1);
  state_0 ~ normal(y[1], .2);
  
  err = state[1] - (mu + state_0);
  err ~ normal(0, sigma);
  
  // state model
  for (t in 2:T){ 
    err <- state[t] - (mu + phi*state[t-1] + theta * err); 
    err ~ normal(0,sigma);
  }
  
  // measurement model
  for(t in 1:T) {
    if(y[t]!=-999) {
      state[t] ~ normal(y[t], .1);
    }
  }
}