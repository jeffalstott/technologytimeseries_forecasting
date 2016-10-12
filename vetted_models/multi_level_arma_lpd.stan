data { 
  int N; // number of observations
  int<lower=1> Tech; // number of technologies
  int n_in_tech[Tech];
  int time[N]; // time within each technology
  int technology[N]; // index for technologies
  vector[N] y; // the log level of technology
  int L; // the AR lag order
  int T2; // the number of periods to predict 
}
transformed data {
  int train[N];
  for(i in 1:N) {
    if(time[i]<=(n_in_tech[technology[i]] - T2)) {
      train[i] = 1;
    } else {
      train[i] = 0;
    }
  }
}
parameters { 
  vector[Tech] mu; // the intercept terms
  matrix[Tech, L] beta; // The AR coefficients
  vector<lower=-1,upper=1>[Tech] theta; // the MA order
  vector<lower=0>[Tech] sigma; // SD of the innovations for a technolgoy
  corr_matrix[3 + L] Omega; // 
  vector[3 + L] mu_parvec;
  vector<lower = 0>[3 + L] tau;
  vector[Tech*T2] y_predict;
} 
transformed parameters {
  matrix[Tech, 3 + L] parvec;
  vector[N] y_compound;
  matrix[N, L] X_compound;
  vector[Tech] K;
  
  for(t in 1:Tech) {
    K[t] <- sqrt( (1+theta[t]^2) * sigma[t]^2 );
  }
  
  parvec <- append_col(mu, append_col(theta, append_col(log(sigma), beta)));
  
  {
    int count;
    count = 0;
    for(i in 1:N) {
      if(train[i]==1) {
        y_compound[i] = y[i];
      } else {
        count = count + 1;
        y_compound[i] = y_predict[count];
      }
    }
  }
  
  X_compound = rep_matrix(0, N, L);
  for(n in 1:N) {
    for(l in 1:L) {
      if(time[n] > L) {
        X_compound[n, l] = y_compound[n-l];
      }
    }
  }
  
}
model {
  vector[N] nu;
  vector[N] err;
  
  for(i in 1:N) {
    if(time[i]<=L) {
      nu[i] = 2*mu[technology[i]];
    } else {
      nu[i] = mu[technology[i]] + X_compound[i]*beta[technology[i]]'  + theta[technology[i]] * err[i-1];
    }
    err[i] = y_compound[i] - nu[i];
  }
  
  // Parameter model
  
  mu_parvec[1] ~ student_t(3, -0.1, 1);
  mu_parvec[2] ~ student_t(3, 0.5, 0.5);
  mu_parvec[3] ~ student_t(3, 0.5, 1);
  mu_parvec[4] ~ student_t(5, 1, 0.5);
  if(L > 1) {
    for(i in 2:L) {
      mu_parvec[3 + i] ~ student_t(5, 0, 0.5);
    }
    
  }
  
  tau ~ cauchy(0, 1);
  Omega ~ lkj_corr(4);
  
  
  for(t in 1:Tech) {
    parvec[t] ~ multi_normal(mu_parvec, quad_form_diag(Omega, tau));
  }
  // likelihood
  for(i in 1:N) {
    if(time[i]>L) {
      err[i] ~ normal(0, sigma[technology[i]]);
    }
  }
}

generated quantities {
  vector[N] lpd;
  vector[Tech] lpd_2;
  
  for(i in 1:N) {
    if(train[i]==1) { 
      lpd[i] = 0.0;
    } else {
      lpd[i] = normal_lpdf(y[i] |  y_compound[i], sigma[technology[i]]);
      if(time[i]==n_in_tech[technology[i]]) {
        lpd_2[technology[i]] = lpd[i];
      }
    }
  }
}
