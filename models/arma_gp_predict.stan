data { 
int<lower=1> T; 
int<lower = 1> T2; // number of periods for prediction
real y[T];
int L; // lags for alpha process
}

parameters { 
vector[T] mu_raw; 
vector[T2] mu_predict;
//real phi;
real<lower=0> sigma; 
real<lower = 0> eta_sq;
real<lower = 0> rho_inv;
real<lower = 0> sigma_alpha;
real mu_hat;
} 

transformed parameters{
//real K;
matrix[T+T2, T + T2] Sigma;
real rho_sq;
vector[T + T2] mu;

for(t in 1:T) mu[t] = mu_raw[t];
for(t in 1:T2) mu[t + T] = mu_predict[t];

rho_sq <- 1/(2*rho_inv^2);
//K <- sqrt( (1+theta^2) * sigma^2 );

for(t1 in 1:(T + T2)) {
  for(t2 in 1:t1) {
    if(t1==t2) {
      Sigma[t1, t2] <- eta_sq + sigma_alpha^2;
    } else {
      if(t1 <= L || t2 <= L) {
        Sigma[t1, t2] <- 0;
        Sigma[t2, t1] <- 0;
      } else {
        {
          real diff;
          diff = 0;
          for(l in 1:L) {
            diff = diff + (mu[t1 - l] - mu[t2 - l])^2;
          }
          Sigma[t1, t2] = eta_sq*exp(-rho_sq*diff);
          Sigma[t2, t1] = Sigma[t1, t2];
        }
      }
    }
  }
}

}

model {
real err; 

//theta ~ normal(0,2); 
sigma ~ cauchy(0,2); 
mu_hat ~ normal(rep_vector(mean(y), T), 0.05);
sigma_alpha ~ normal(0, 0.5);
eta_sq ~ normal(0, 0.1);
rho_inv ~ cauchy(5, 2);

mu ~ multi_normal(rep_vector(mu_hat, T+T2),Sigma); 

y ~ normal(mu_raw, sigma);
}
generated quantities {
  vector[T2] y_predict;
  for(t in 1:T2) {
    y_predict[t] = normal_rng(mu_predict[t], sigma);
  }
}