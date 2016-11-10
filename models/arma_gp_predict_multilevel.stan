data { 
int<lower=1> N; // number of observations
int N_tech; // number of technologies
int n_in_technology[N]; // indicator telling us how many observations in the given technology
int technology[N]; // The technology index
int time[N]; // number of years into the technology
int start[N_tech];
int end[N_tech];
vector[N] y; // the full (train and test set) indicators
vector[N] meanvec;
int T2; // number of observations in test set
int L; // lags for mu process
}
transformed data  {
  int train_set[N];
  int test_set[N];
  
  for (i in 1:N) {
    if(time[i] <= (n_in_technology[i] - T2)) {
      train_set[i] = 1;
      test_set[i] = 0;
    } else { 
      train_set[i] = 0;
      test_set[i] = 1;
      }
  }
}
parameters { 
vector[N] mu; 
//real phi;
vector<lower=0>[N_tech] sigma; 
vector<lower=0>[N_tech] eta_sq;
vector<lower=0>[N_tech] rho_inv;
vector<lower=0>[N_tech] sigma_alpha;
} 

transformed parameters{
//real K;
matrix[N, N] Sigma;
vector[N_tech] rho_sq;

for(n in 1:N_tech) {
  rho_sq[n] = 1/(2*rho_inv[n]^2);
}

// Use 0 for all off-chunks
Sigma = rep_matrix(0, N, N);

// Now fill out the blocks

for(i in 1:N_tech) {
  
  int count1;
  count1 = 1;
  for(t1 in start[i]:end[i]) {
    int count2;
    count2 = 1;
    for(t2 in start[i]:t1) {
      
      if(t1==t2) {
        Sigma[t1, t2] = eta_sq[i] + sigma_alpha[i]^2;
      } else {
        if(count1 <= L || count2 <= L) {
          Sigma[t1, t2] = 0;
          Sigma[t2, t1] = 0;
        } else {
          {
            real diff;
            diff = 0;
            for(l in 1:L) {
              diff = diff + (mu[t1 - l] - mu[t2 - l])^2;
            }
            
            Sigma[t1, t2] = eta_sq[i]*exp(-rho_sq[i]*diff);
            Sigma[t2, t1] = Sigma[t1, t2];
          }
        }
      }
      count2 = count2 + 1;
    }
    count1 = count1 + 1;
  }
}
}

model {

//theta ~ normal(0,2); 
sigma ~ normal(0,0.2); 
sigma_alpha ~ normal(0, 0.5);
eta_sq ~ normal(0, 0.1);
rho_inv ~ cauchy(5, 2);

mu ~ multi_normal(meanvec,Sigma); 

for(i in 1:N) {
  if(train_set[i]==1) { 
    y[i] ~ normal(mu[i], sigma[technology[i]]);
  }
}

}
generated quantities {
  vector[N] y_predict;
  for(t in 1:N) {
    if(test_set[t]==1) {
      y_predict[t] = normal_rng(mu[t], sigma[technology[t]]);
    } else {
      y_predict[t] = y[t];
    }
  }
  
}