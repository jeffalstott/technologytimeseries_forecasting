data { 
int<lower=1> N; // number of observations
int N_tech; // number of technologies
int n_in_technology[N_tech]; // indicator telling us how many observations in the given technology
int technology[N]; // The technology index
int time[N]; // number of years into the technology
int start[N_tech];
int end[N_tech];
vector[N] y; // the full (train and test set) indicators
int T2; // number of observations in test set
int L; // lags for mu process
int n_non_zero;
}
transformed data  {
  int train_set[N];
  int test_set[N];
  int non_zero_rows_u[N + 1];
  int non_zero_cols_v[n_non_zero];
  
  for (i in 1:N) {
    if(time[i] <= (n_in_technology[technology[i]] - T2)) {
      train_set[i] = 1;
      test_set[i] = 0;
    } else { 
      train_set[i] = 0;
      test_set[i] = 1;
      }
  }
  {
    matrix[N, N] Sigma;
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
        Sigma[t1, t2] = 1;
      } else {
        if(count1 <= L || count2 <= L) {
          Sigma[t1, t2] = 0;
          Sigma[t2, t1] = 0;
        } else {
          Sigma[t1, t2] = 1;
          Sigma[t2, t1] = 1;
          
        }
      }
      count2 = count2 + 1;
    }
    count1 = count1 + 1;
  }
}
  non_zero_rows_u = csr_extract_u(Sigma);
  non_zero_cols_v = csr_extract_v(Sigma);
  
  }
}
parameters { 
vector[N] mu; 
//real phi;
vector<lower=0>[N_tech] sigma; 
vector<lower=0>[N_tech] eta_sq;
vector<lower=0>[N_tech] rho_inv;
vector<lower=0>[N_tech] sigma_alpha;
vector[N_tech] meanhat;
vector[5] hypermeans;
vector<lower = 0>[5] hypertau;
corr_matrix[5] Omega;


} 

transformed parameters{
//real K;
vector[n_non_zero] non_zero_covariances;
vector[N_tech] rho_sq;
matrix[N_tech, 5] hyper_pars;
vector[N] meanvec;

for(i in 1:N) meanvec[i] = meanhat[technology[i]];


hyper_pars = append_col(log(sigma), append_col(log(eta_sq), append_col(log(rho_inv), append_col(log(sigma_alpha), meanhat))));

for(n in 1:N_tech) {
  rho_sq[n] = 1/(2*rho_inv[n]^2);
}


{
  matrix[N, N] Sigma;
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
non_zero_covariances = csr_extract_w(Sigma);
}

}

model {
//hyper_pars = append_col(log(sigma), append_col(log(eta_sq), append_col(log(rho_inv), log(sigma_alpha))));
hypertau ~ cauchy(0, 1);
Omega ~ lkj_corr(4);

sigma ~ normal(0,0.2); 
eta_sq ~ normal(0, 0.1);
rho_inv ~ cauchy(5, 2);
sigma_alpha ~ normal(0, 0.5);
meanhat ~ cauchy(-0.1, 0.5);


for(i in 1:N_tech) {
  hyper_pars[i] ~ multi_normal(hypermeans, quad_form_diag(Omega, hypertau));
}

mu ~ multi_normal(meanvec,csr_to_dense_matrix(N, N, non_zero_covariances, non_zero_cols_v, non_zero_rows_u)); 

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