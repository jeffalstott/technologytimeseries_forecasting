data { 
int<lower=1> T; 
real y[T];
}

parameters { 
real mu; 
//real phi;
real<lower=-1,upper=1> theta; 
real<lower=0> sigma; 
} 

transformed parameters{
real K;
K <- sqrt( (1+theta^2) * sigma^2 );
}

model {
real err; 
mu ~ normal(0,10); 
theta ~ uniform(-1,1); //normal(0,2); 
sigma ~ cauchy(0,5); 
err <- y[1] - 2*mu; //phi = 1
err ~ normal(0,sigma); 
for (t in 2:T){ 
    err <- y[t] - (mu + y[t-1] + theta * err); 
    err ~ normal(0,sigma);
    }
}