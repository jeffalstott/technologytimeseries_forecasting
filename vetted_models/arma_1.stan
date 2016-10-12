data { 
int<lower=1> T; 
vector[T] y;
}

parameters { 
real mu; 
//real phi;
real<lower=-1,upper=1> theta; 
real<lower=0> sigma; 
real y_0; // initial value
} 

transformed parameters{
real K;
K <- sqrt( (1+theta^2) * sigma^2 );
}

model {
real err; 
mu ~ normal(0,1); 
theta ~ normal(0,1); 
sigma ~ cauchy(0,2); 
y_0 ~ normal(y[1], 1);

err = y[1] - (mu + y_0);
err ~ normal(0,sigma); 

for (t in 2:T){ 
    err <- y[t] - (mu + y[t-1] + theta * err); 
    err ~ normal(0,sigma);
    }
}
generated quantities {
  vector[T] lpd;
  real err;
  err = y[1] - (mu + y_0);
  lpd[1] = normal_lpdf(y[1] | y[1] + theta*err, sigma);
  for(t in 2:T) {
    err = y[t] - (mu + y[t-1] + theta * err); 
    lpd[t] = normal_lpdf(y[t] | mu + y[t-1] + theta*err, sigma);
  }
  
}