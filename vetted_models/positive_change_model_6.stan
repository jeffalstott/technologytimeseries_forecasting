data {
  int T;
  int N_missing;
  int N_missing_r;
  int fill_in_r[T];
  vector[T] Y;
}
parameters {
  vector[N_missing] yhat;
  vector[N_missing_r] r;
  real<lower = 0> r_init;
  real<lower = 0, upper = 1> theta;
  real<lower = 0> mu;
  real<lower = 0> sigma;
}
transformed parameters {
  vector[T] latent_Y;
  vector[T] latent_r;
 {
   int count;
   int count2;
   
   count = 0;
   count2 = 0;
   
   for(t in 1:T) {
     
     
     if(fill_in_r[t]==1) {
       count2 = count2+1;
       latent_r[t] = r[count2];
     } else {
       if(t==1) {
         latent_r[t] = r_init;
       } else {
         latent_r[t] = Y[t] - Y[t-1];
       }
     }
     
     if(Y[t]> -900) {
       latent_Y[t] = Y[t];
     } else {
       count = count + 1;
       latent_Y[t] = yhat[count];
     }
   }
 } 
  
}

model {
  
  // some priors
  mu ~ normal(0, 1);
  sigma ~ cauchy(0, 2);
  theta ~ beta(1.1, 1.1);
  r ~ normal(mu, sigma);
  r_init ~ normal(0, 1);
  
  // likelihood
  for(t in 2:T) {
    real mu_1;
    real mu_2;
    
    mu_1 = latent_Y[t-1];
    mu_2 = latent_Y[t-1] + latent_r[t];
    
    target += log_mix(theta, normal_lpdf(latent_Y[t] |  mu_1, 0.01), 
                      normal_lpdf(latent_Y[t] | mu_2, 0.01));
  }
}