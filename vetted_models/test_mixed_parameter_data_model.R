library(rstan)
options(mc.cores = parallel::detectCores())


T <- 200
mu <- .3
sigma <- .2
theta <- .6
missing_prob <- .3
r <- rnorm(1000, mu, sigma)
r <- r[r>0]
r <- r[1:T]

latent_Y <- NULL
latent_Y[1] <- 1

for (t in 2:T) {
  mu_1 <- latent_Y[t-1]
  mu_2 <- latent_Y[t-1] + r[t-1]
  z <- rbinom(1, 1, theta)
  if(z==1) {
    latent_Y[t] <- rnorm(1, mu_1, 0.01)
  } else {
    latent_Y[t] <- rnorm(1, mu_2, 0.01)
  }
}


# missing values
Y <- NULL
Y[1] <- latent_Y[1]

for(t in 2:T) {
  zz <- rbinom(1, 1, missing_prob)
  if(zz==1) {
    Y[t] <- -999
  } else {
    Y[t] <- latent_Y[t]
  }
}

fill_in_r <- Y==-999 | c(NA, Y[-T])==-999
fill_in_r[1] <- F

data.frame(Y,  fill_in_r)

compiled_model <- stan_model("vetted_models/positive_change_model_6.stan")

test_model <- sampling(compiled_model, data = list(T = T, 
                                                   N_missing = sum(Y==-999),
                                                   N_missing_r = sum(fill_in_r),
                                                   fill_in_r = fill_in_r,
                                                   Y = Y), 
                       iter = 600, cores =4)
