__author__ = 'thor'

model_code = """
// Mixture of two binomials
data {
    int<lower=1> nExperiments; // number of data points
    int<lower=0> nTrials[nExperiments];
    int<lower=0> nSuccess[nExperiments];
}

parameters {
    simplex[2] theta; // mixing proportions
    real<lower=0,upper=1> latentProb[2]; // locations of mixture components
}

model {
    real ps[2];  // temp for log component densities
    for (k in 1:2) {
        latentProb[k] ~ beta(1,1);
    }
    for (n in 1:nExperiments) {
        for (k in 1:2) {
            ps[k] <- binomial_log(nSuccess[n], nTrials[n], latentProb[k]);
        }
        increment_log_prob(log_sum_exp(ps));
    }
}
"""
sm = pystan.StanModel(model_code=model_code)

nExperiments = 100
K = 2
theta = rand(K); theta = theta / K;

data = ms.stats.dgen.bin.binomial_mixture(
    npts=nExperiments,
    n_trials=[1, 20],
    n_components=2,
    include_component_idx=True,
    include_component_prob=True,
    n_trials_col='nTrials',
    n_success_col='nSuccess'
)

sdata = {
    'nExperiments': len(data),
    'nTrials': data['nTrials'],
    'nSuccess': data['nSuccess']
}

def initialisation():
    latentProb = rand(K);
    theta = rand(K); theta /= sum(theta)
    return {'theta': [0.5]*K, 'latentProb': [0.5]*K}

sm.sampling(
            model_code=model_code,
            data=sdata,
            init=initialisation,
            iter=100,
            refresh=10,
            chains=4)