import scipy.stats as stats
import numpy as np
import pandas as pd

# Define your dataset
data = pd.read_csv(r"C:\Users\Alicia BASSIERE\OneDrive - GENES\Documents\Paper 01 - DIPU\outputs\Cas V6\npv_nopolicy.csv", index_col=0)

# Define the negative log-likelihood function for the beta distribution
def neg_log_likelihood(params, data):
    alpha, beta = params
    return -np.sum(stats.beta.logpdf(data, alpha, beta))

# Use the minimize function from scipy.optimize to find the maximum likelihood estimate
result = stats.optimize.minimize(neg_log_likelihood, [1, 1], args=(data,))
alpha_mle, beta_mle = result.x

# Print the estimated parameters
print(f"Estimated alpha: {alpha_mle}")
print(f"Estimated beta: {beta_mle}")

